"""
Module to launch pretraining job on synthetic motion dataset.
"""

import logging
import random
import shutil
from typing import Type
import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)

from src.utils.comet import get_experiment_key
from src.dataset.pretraining.pretraining_dataset import PretrainingDataModule
from src.training.eval import SaveBestCheckpoint, get_correlations
from src.training.pretrain_logic import PretrainingTask
from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME
from src.utils.log import get_run_dir
from src.utils.mcdropout import pretrain_mcdropout
from src.utils.task import EnsureOneProcess, label_from_task, str_to_task

torch.set_float32_matmul_precision("high")


def launch_pretrain(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    model: str,
    run_num: int,
    seed: int | None,
    use_cutout: bool,
    task: str,
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
        use_cutout(bool): flag to use cutout in model training
        task(str): Pretraining task to use
    """

    run_name = f"pretraining-{task}-{model}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=COMET_API_KEY,
        project_name=PROJECT_NAME,
        experiment_name=run_name,
        experiment_key=get_experiment_key("mrart", PROJECT_NAME, run_name),
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams(
        {
            "seed": seed,
            "model": model,
            "run_num": run_num,
            "use_cutout": use_cutout,
            "task": task,
        }
    )
    comet_logger.experiment.log_code(file_name="src/commands/pretrainer.py")
    logging.info("Run dir path is : %s", run_dir)

    task_class: Type[PretrainingTask] = str_to_task(task)

    net = task_class(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        use_cutout=use_cutout,
    )

    checkpoint = SaveBestCheckpoint(monitor="r2_score", mode="max")

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
                patience=30,
                check_on_train_epoch_end=False,
                verbose=True,
            ),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(net, datamodule=PretrainingDataModule(batch_size, task))

    with EnsureOneProcess(trainer):
        best_net = task_class.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path
        )
        logging.info("Running correlation on pretrain")
        get_correlations(best_net, comet_logger.experiment)

        logging.info("Running dropout on pretrain")
        pretrain_mcdropout(
            best_net,
            trainer.val_dataloaders,
            comet_logger.experiment,
            label=label_from_task(task),
        )

        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
