"""
Module to launch the training process from scratch (no pretraining)
Used as baseline training technic
"""

import tempfile
import random
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.dataset.mrart.mrart_dataset import MRArtDataModule
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.config import IM_SHAPE
from src.training.callback import FinetuneCallback
from src.training.lightning_logic import (
    AMPSCZScratchTask,
    MRArtScratchTask,
    TrainScratchTask,
)


def launch_train_from_scratch(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    dataset: str,
    model: str,
    run_num: int,
    seed: int | None,
    narval: bool,
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
        narval (bool): flag to run on narval computers
    """
    assert dataset in ("MRART", "AMPSCZ"), "Dataset does not exist"

    task: TrainScratchTask = None
    datamodule: lightning.LightningDataModule = None
    if dataset == "MRART":
        datamodule = MRArtDataModule
        task = MRArtScratchTask
    elif dataset == "AMPSCZ":
        datamodule = AMPSCZDataModule
        task = AMPSCZScratchTask

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name=f"baseline-{dataset}",
        experiment_name=f"{model}-{run_num}",
    )
    if seed is None:
        seed = random.randint(1, 10000)
    comet_logger.log_hyperparams({"seed": seed, "model": model, "run_num": run_num})
    comet_logger.experiment.log_code(file_name="src/commands/base_trainer.py")

    tempdir = tempfile.TemporaryDirectory()

    net = task(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            FinetuneCallback(monitor="val_balanced_accuracy", mode="max"),
        ],
    )

    trainer.fit(net, datamodule=datamodule(narval, batch_size))
