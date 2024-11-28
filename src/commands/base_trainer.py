"""
Module to launch the training process from scratch (no pretraining)
Used as baseline training technic
"""

import logging
import os
import random
import shutil
import tempfile

import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src import config
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.training.scratch_logic import AMPSCZScratchTask
from src.utils.task import EnsureOneProcess


def launch_train_from_scratch(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    weight_decay: float,
    run_num: int,
    seed: int | None,
):
    """Start training from scratch

    Args:
        learning_rate (float): training learning rate
        dropout_rate (float): dropout rate before final layer
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
    """
    project_name = f"baseline-AMPSCZ"
    run_name = f"scratch-SFCN-{run_num}"
    report_name = f"{run_name}-AMPSCZ"
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", "scratch", report_name)
    if os.path.exists(save_model_path):
        shutil.rmtree(save_model_path)
    os.makedirs(save_model_path, exist_ok=True)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=project_name,
        experiment_name=f"SFCN-{run_num}",
    )
    if seed is None:
        seed = random.randint(1, 10000)
    comet_logger.log_hyperparams({"seed": seed, "run_num": run_num})
    comet_logger.experiment.log_code(file_name="src/commands/base_trainer.py")
    comet_logger.experiment.log_code(file_name="src/training/scratch_logic.py")

    tempdir = tempfile.TemporaryDirectory()

    net = AMPSCZScratchTask(
        im_shape=config.IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    checkpoint = ModelCheckpoint(monitor="val_balanced_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=1,
        accelerator="gpu",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=100),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    data = AMPSCZDataModule(batch_size)
    trainer.fit(net, datamodule=data)

    with EnsureOneProcess(trainer):
        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=net.model.__class__.__name__,
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
