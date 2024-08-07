"""
Module to launch pretraining job on synthetic motion dataset.
"""

import logging
import random

import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)

from src.utils.comet import get_experiment_key
from src.dataset.pretraining.pretraining_dataset import PretrainingDataModule
from src.training.callback import PretrainCallback
from src.training.lightning_logic import PretrainingTask
from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME
from src.utils.log import get_run_dir


def launch_pretrain(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    model: str,
    run_num: int,
    seed: int | None,
    narval: bool,
    use_cutout: bool,
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
        narval (bool): flag to run on narval computers
        use_cutout(bool): flag to use cutout in model training
    """

    run_name = f"pretraining-{model}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name, narval)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=COMET_API_KEY,
        project_name=PROJECT_NAME,
        experiment_name=run_name,
        experiment_key=get_experiment_key(
            COMET_API_KEY, "mrart", PROJECT_NAME, run_name
        ),
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams(
        {"seed": seed, "model": model, "run_num": run_num, "use_cutout": use_cutout}
    )
    comet_logger.experiment.log_code(file_name="src/commands/pretrainer.py")
    logging.info("Run dir path is : %s", run_dir)
    net = PretrainingTask(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        use_cutout=use_cutout,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=2,
        strategy="ddp",
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=40,
                check_on_train_epoch_end=False,
                verbose=True,
            ),
            PretrainCallback(monitor="r2_score", mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(net, datamodule=PretrainingDataModule(narval, batch_size))
