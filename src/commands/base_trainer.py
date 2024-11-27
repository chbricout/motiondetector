"""
Module to launch the training process from scratch (no pretraining)
Used as baseline training technic
"""

import logging
import os
import random
import shutil
import tempfile
from typing import Type

import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.tuner import Tuner

from src import config
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.dataset.mrart.mrart_dataset import MRArtDataModule, UnbalancedMRArtDataModule
from src.training.scratch_logic import (
    AMPSCZScratchTask,
    MRArtScratchTask,
    TrainScratchTask,
)
from src.utils.log import get_run_dir
from src.utils.mcdropout import transfer_mcdropout
from src.utils.task import EnsureOneProcess


def launch_train_from_scratch(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    weight_decay: float,
    dataset: str,
    model: str,
    run_num: int,
    seed: int | None,
):
    """Start training from scratch

    Args:
        learning_rate (float): training learning rate
        dropout_rate (float): dropout rate before final layer
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        dataset (str): dataset to train on "AMPSCZ" or "MRART"
        model (str): model to train
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
    """
    assert dataset in ("MRART", "AMPSCZ", "UNBALANCED-MRART"), "Dataset does not exist"
    project_name = (
        f"baseline-{dataset}" if dataset != "UNBALANCED-MRART" else "unbalanced-mrart"
    )

    run_name = f"scratch-{model}-{run_num}"
    report_name = f"{run_name}-{dataset}"
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", "scratch", report_name)
    if os.path.exists(save_model_path):
        shutil.rmtree(save_model_path)
    os.makedirs(save_model_path, exist_ok=True)

    task: Type[TrainScratchTask] = None
    datamodule: Type[lightning.LightningDataModule] = None
    if dataset == "MRART":
        datamodule = MRArtDataModule
        task = MRArtScratchTask
    elif dataset == "UNBALANCED-MRART":
        datamodule = UnbalancedMRArtDataModule
        task = MRArtScratchTask
    elif dataset == "AMPSCZ":
        datamodule = AMPSCZDataModule
        task = MRArtScratchTask

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=project_name,
        experiment_name=f"{model}-{run_num}",
    )
    if seed is None:
        seed = random.randint(1, 10000)
    comet_logger.log_hyperparams({"seed": seed, "model": model, "run_num": run_num})
    comet_logger.experiment.log_code(file_name="src/commands/base_trainer.py")
    comet_logger.experiment.log_code(file_name="src/training/scratch_logic.py")

    tempdir = tempfile.TemporaryDirectory()

    net = task(
        model_class=model,
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
        # precision="16-mixed",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=100),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    data = datamodule(batch_size)
    trainer.fit(net, datamodule=data)

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
