"""
Module to launch finetuning job on pretrained model.
"""

import logging
import os
import random
import shutil

import lightning
import lightning.pytorch.loggers
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src import config
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.network.archi import Encoder
from src.training.transfer_logic import AMPSCZTransferTask
from src.utils.log import get_run_dir
from src.utils.task import EnsureOneProcess, load_pretrain_from_ckpt


def launch_transfer(
    pretrain_path: str,
    learning_rate: float,
    max_epochs: int,
    dropout_rate: float,
    batch_size: int,
    weight_decay: float,
    num_layers: int,
    run_num: int,
    seed: int | None,
):
    """Launch the finetuning process

    Args:
        estimator (str) : pretraining estimator to transfer on
        learning_rate (float): training learning rate
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
    """
    project_name = f"transfer-AMPSCZ"

    run_name = f"AMPSCZ-SFCN-MOTION-{run_num}"
    run_dir = get_run_dir(project_name, run_name)
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", "transfer", run_name)
    os.makedirs(save_model_path, exist_ok=True)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=project_name,
        experiment_name=run_name,
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams({"seed": seed, "run_num": run_num})
    comet_logger.experiment.log_code(file_name="src/commands/transfer.py")
    logging.info("Run dir path is : %s", run_dir)

    pretrained = load_pretrain_from_ckpt(pretrain_path)

    encoding_model: Encoder = pretrained.model.encoder

    net = AMPSCZTransferTask(
        input_size=encoding_model.latent_shape,
        pretrained=pretrained.model,
        lr=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    checkpoint = ModelCheckpoint(monitor="val_balanced_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=1,
        accelerator="gpu",
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=20,
                verbose=True,
            ),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    datamod = AMPSCZDataModule(batch_size, encoding_model)
    trainer.fit(net, datamodule=datamod)

    with EnsureOneProcess(trainer):
        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name="SFCNModel",
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
