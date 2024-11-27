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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch._dynamo.eval_frame import OptimizedModule

from src import config
from src.dataset.pretraining.pretraining_dataset import PretrainingDataModule
from src.utils import confidence as conf
from src.utils import mcdropout
from src.utils.comet import log_figure_comet
from src.utils.log import get_run_dir
from src.utils.task import (
    EnsureOneProcess,
    label_from_task,
    load_pretrain_from_ckpt,
    str_to_task,
)


def launch_continual(
    pretrain_path: str,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    task: str,
    run_num: int,
    seed: int | None,
):
    """Launch the continual process

    Args:
    """
    project_name = "continual-synthetic"
    task = "CONTINUAL" + "-" + task
    model, pretrain_task, _ = os.path.basename(pretrain_path).split("-")
    run_name = f"continual-{model}-{task}-{run_num}"
    run_dir = get_run_dir(project_name, run_name)
    os.makedirs("model_report", exist_ok=True)
    save_model_path = os.path.join("model_report", run_name)
    os.makedirs(save_model_path, exist_ok=True)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=project_name,
        experiment_name=run_name,
    )

    if seed is None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams({"seed": seed, "model": model, "run_num": run_num})
    comet_logger.experiment.log_code(file_name="src/commands/continual_trainer.py")
    logging.info("Run dir path is : %s", run_dir)

    pretrained, _, model_str = load_pretrain_from_ckpt(pretrain_path)
    pretrained_model = pretrained.model
    if isinstance(pretrained_model, OptimizedModule):
        pretrained_model = pretrained_model._orig_mod
    task_cls = str_to_task(task)

    net = task_cls(
        pretrained=pretrained_model,
        lr=learning_rate,
        batch_size=batch_size,
    )

    checkpoint = ModelCheckpoint(monitor="balanced_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=4 if config.IS_NARVAL else [1],
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=10,
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

    trainer.fit(net, datamodule=PretrainingDataModule(batch_size, task))

    with EnsureOneProcess(trainer):
        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=net.model.__class__.__name__,
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)

        best_net = task_cls.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path
        )

        logging.info("Running dropout on pretrain")

        drop_res = mcdropout.predict_mcdropout(
            best_net, net.val_dataloader(), n_preds=10, label=label_from_task(task)
        )
        dropout_df = mcdropout.finetune_pred_to_df(*drop_res)
        conf_df = conf.confidence_finetune(dropout_df)
        comet_logger.experiment.log_table("Dropout_df", dropout_df)
        fig = conf.plot_confidence(
            conf_df=conf_df,
            threshold_label="threshold_confidence",
            metric_label="balanced_accuracy",
            threshold_axis=("Threshold Confidence"),
            metric_axis="Balanced Accuracy",
        )
        log_figure_comet(fig, "Confidence", comet_logger.experiment)

        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
