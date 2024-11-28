"""
Module to launch pretraining job on synthetic motion dataset.
"""

import logging
import os
import random
import shutil
from typing import Type

import comet_ml
import lightning
import lightning.pytorch.loggers
import torch
import torch.distributed
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME
from src.dataset.pretraining.pretraining_dataset import PretrainingDataModule
from src.training.pretrain_logic import MotionPretrainingTask
from src.utils.log import get_run_dir
from src.utils.task import EnsureOneProcess

torch.set_float32_matmul_precision("high")


def launch_pretrain(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    model: str,
    run_num: int,
    seed: int | None,
):
    """Launch the pretraining process

    Args:
        learning_rate (float): training learning rate
        dropout_rate (float): dropout rate before final layer
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        model (str): model to train
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
    """

    run_name = f"pretraining-{model}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name)
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", run_name)
    os.makedirs(save_model_path, exist_ok=True)
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=COMET_API_KEY,
        project_name=PROJECT_NAME,
        experiment_name=run_name,
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams(
        {
            "seed": seed,
            "model": model,
            "run_num": run_num,
        }
    )
    comet_logger.experiment.log_code(file_name="src/commands/pretrainer.py")
    logging.info("Run dir path is : %s", run_dir)

    net = MotionPretrainingTask(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

    checkpoint = ModelCheckpoint(monitor="r2_score", mode="max")

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=200,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=15,
                check_on_train_epoch_end=False,
                verbose=True,
            ),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(net, datamodule=PretrainingDataModule(batch_size))

    with EnsureOneProcess(trainer):

        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=net.model.__class__.__name__,
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)

        best_net = MotionPretrainingTask.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path
        )

        logging.info("Running dropout on pretrain")
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
