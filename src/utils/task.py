"""Module to define any utility function to use tasks"""

import sys
from os import path
from typing import Type

import torch
from lightning import Trainer

from src import config
from src.training.pretrain_logic import (
    BinaryPretrainingTask,
    ContinualMotionPretrainingTask,
    ContinualPretrainingTask,
    ContinualSSIMPretrainingTask,
    MotionPretrainingTask,
    PretrainingTask,
    SSIMPretrainingTask,
)


class EnsureOneProcess:
    """Context to ensure running on one process"""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __enter__(self):
        self.trainer.strategy.barrier()
        if not self.trainer.is_global_zero:
            sys.exit(0)

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


def label_from_task(task: str) -> str:
    """Retrieve label column name in dataframe from Pretrain task

    Args:
        task (str): Task to pretrain on

    Returns:
        str: label for dataset
    """
    task_class = str_to_task(task)
    return label_from_task_class(task_class)


def label_from_task_class(task_class: Type[PretrainingTask]) -> str:
    """Retrieve label column name in dataframe from Pretrain task

    Args:
        task (str): Task to pretrain on

    Returns:
        str: label for dataset
    """
    label = ""
    if task_class == MotionPretrainingTask:
        label = "motion_mm"
    elif task_class == SSIMPretrainingTask:
        label = "ssim_loss"
    elif task_class == BinaryPretrainingTask:
        label = "motion_binary"
    elif task_class == ContinualSSIMPretrainingTask:
        label = "three_ssim"
    elif task_class == ContinualMotionPretrainingTask:
        label = "three_motion"
    return label


def str_to_task(task_str: str) -> Type[PretrainingTask]:
    """Retrieve task class from string

    Args:
        task_str (str): Task as a string

    Returns:
        Type[PretrainingTask]: Class of task
    """
    task_class: Type[PretrainingTask] = None
    if task_str == "MOTION":
        task_class = MotionPretrainingTask
    elif task_str == "SSIM":
        task_class = SSIMPretrainingTask
    elif task_str == "BINARY":
        task_class = BinaryPretrainingTask
    elif task_str == "CONTINUAL-MOTION":
        task_class = ContinualMotionPretrainingTask
    elif task_str == "CONTINUAL-SSIM":
        task_class = ContinualSSIMPretrainingTask
    assert not task_class is None, f"Error, task {task_str} doesnt exists"
    return task_class


def load_pretrain_from_ckpt(ckpt_path: str):
    model_str, task = parse_transfer_path(ckpt_path)
    task_class = str_to_task(task)
    checkpoint = torch.load(ckpt_path)
    checkpoint["state_dict"].pop(
        "label_loss.pos_weight", None
    )  # Safely removes the key
    if issubclass(task_class, ContinualPretrainingTask):
        module = task_class.load_from_checkpoint(ckpt_path)
    else:
        module = task_class(model_str, config.IM_SHAPE)
        module.load_state_dict(checkpoint["state_dict"])
    return module, task, model_str


def parse_transfer_path(ckpt_path: str):
    base_name = path.basename(ckpt_path)
    model_str, task, *_ = base_name.split("-")
    task = task.replace("_", "-")
    return model_str, task
