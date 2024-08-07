"""
Module to launch finetuning job on pretrained model.
"""

import logging
import random

import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.dataset.mrart.mrart_dataset import MRArtDataModule
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.config import IM_SHAPE, PROJECT_NAME
from src.training.callback import FinetuneCallback
from src.training.lightning_logic import (
    MRArtFinetuningTask,
    AMPSCZFinetuningTask,
    TrainScratchTask,
)
from src.utils.comet import get_pretrain_task
from src.utils.log import get_run_dir


def launch_finetune(
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    dataset: str,
    model: str,
    run_num: int,
    seed: int | None,
    narval: bool,
):
    """Launch the finetuning process

    Args:
        learning_rate (float): training learning rate
        max_epochs (int): max number of epoch to train for
        batch_size (int): batch size (on one GPU)
        dataset (str): dataset to train on "AMPSCZ" or "MRART"
        model (str): model to train
        run_num (int): array id for slurm job when running multiple seeds
        seed (int | None): random seed to run on
        narval (bool): flag to run on narval computers
    """
    assert dataset in ("MRART", "AMPSCZ"), "Dataset does not exist"

    run_name = f"finetune-{dataset}-{model}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name, narval)

    task: TrainScratchTask = None
    datamodule: lightning.LightningDataModule = None
    if dataset == "MRART":
        datamodule = MRArtDataModule
        task = MRArtFinetuningTask
    elif dataset == "AMPSCZ":
        datamodule = AMPSCZDataModule
        task = AMPSCZFinetuningTask

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name=PROJECT_NAME,
        experiment_name=run_name,
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams(
        {"seed": seed, "model": model, "dataset": dataset, "run_num": run_num}
    )
    comet_logger.experiment.log_code(file_name="src/commands/finetune.py")
    logging.info("Run dir path is : %s", run_dir)

    pretrained = get_pretrain_task(model, run_num, PROJECT_NAME)
    net = task(pretrained_model=pretrained.model, im_shape=IM_SHAPE, lr=learning_rate)

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=100),
            FinetuneCallback(monitor="val_balanced_accuracy", mode="max"),
        ],
    )

    trainer.fit(net, datamodule=datamodule(narval, batch_size))
