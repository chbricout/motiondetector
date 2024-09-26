"""
Module to launch finetuning job on pretrained model.
"""

import logging
import os
import random
import shutil
from typing import Type

import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src import config
from src.dataset.mrart.mrart_dataset import MRArtDataModule
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.config import  PROJECT_NAME
from src.network.archi import Encoder
from src.training.pretrain_logic import PretrainingTask
from src.training.transfer_logic import (
    TransferTask,
    MrArtTransferTask,
    AMPSCZTransferTask,
)
from src.utils.comet import get_pretrain_task
from src.utils.log import get_run_dir
from src.utils.mcdropout import transfer_mcdropout
from src.utils.task import EnsureOneProcess, str_to_task


def launch_transfer(
    pretrain_path: str,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    dataset: str,
    run_num: int,
    seed: int | None,
):
    """Launch the finetuning process

    Args:
        estimator (str) : pretraining estimator to transfer on
        learning_rate (float): training learning rate
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        dataset (str): dataset to train on "AMPSCZ" or "MRART"
        model (str): model to train()
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
    """
    assert dataset in ("MRART", "AMPSCZ"), "Dataset does not exist"
    model, pretrain_task, _ = os.path.basename(pretrain_path).split('-')
    run_name = f"transfer-{dataset}-{model}-{pretrain_task}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name)
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", run_name)
    os.makedirs(save_model_path, exist_ok=True)

    task: Type[TransferTask] = None
    datamodule: lightning.LightningDataModule = None
    if dataset == "MRART":
        datamodule = MRArtDataModule
        task = MrArtTransferTask
    elif dataset == "AMPSCZ":
        datamodule = AMPSCZDataModule
        task = AMPSCZTransferTask

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=PROJECT_NAME,
        experiment_name=run_name,
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams(
        {"seed": seed, "model": model, "dataset": dataset, "run_num": run_num}
    )
    comet_logger.experiment.log_code(file_name="src/commands/transfer.py")
    logging.info("Run dir path is : %s", run_dir)

    task_class: Type[PretrainingTask] = str_to_task(pretrain_task)
    pretrained = task_class.load_from_checkpoint(checkpoint_path=pretrain_path)
    
    encoding_model: Encoder = pretrained.model.encoder
    net = task(
        input_size=encoding_model.latent_shape,
        encoder=encoding_model,
        lr=learning_rate,
        batch_size=batch_size,
    )

    checkpoint = ModelCheckpoint(monitor="val_balanced_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=300,
                verbose=True,
            ),
            checkpoint,
        ],
    )

    trainer.fit(net, datamodule=datamodule(batch_size, encoding_model))

    with EnsureOneProcess(trainer):
        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=net.model.__class__.__name__,
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)

        best_net = task.load_from_checkpoint(checkpoint_path=checkpoint.best_model_path)

        transfer_mcdropout(best_net, trainer.val_dataloaders, comet_logger.experiment)
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
