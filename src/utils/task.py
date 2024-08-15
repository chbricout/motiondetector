from typing import Type

from lightning import Trainer
from src.training.lightning_logic import (
    BinaryPretrainingTask,
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
            exit(0)

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
    return label


def str_to_task(task_str: str) -> Type[PretrainingTask]:
    task_class: Type[PretrainingTask] = None
    if task_str == "MOTION":
        task_class = MotionPretrainingTask
    elif task_str == "SSIM":
        task_class = SSIMPretrainingTask
    elif task_str == "BINARY":
        task_class = BinaryPretrainingTask
    assert not task_class is None, "Error, task doesnt exists"
    return task_class
