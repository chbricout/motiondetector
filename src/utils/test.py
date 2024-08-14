"""Module to provide utility function for test purposes only !"""

from typing import Type

from lightning import LightningModule
import torch
from src import config
from src.network.utils import parse_model
from src.training.lightning_logic import (
    FinetuningTask,
    MotionPretrainingTask,
    SSIMPretrainingTask,
    MRArtFinetuningTask,
    MRArtScratchTask,
)
from src.transforms.load import ToSoftLabel


def parse_module(
    task_class: Type[FinetuningTask | LightningModule], model: str
) -> LightningModule:
    """Return correct lightning module with default init depending on the type of task
    for test purposes

    Args:
        task_class (Type[FinetuningTask  |  LightningModule]): Class of the task
        model (str): model name

    Returns:
        LightningModule: initialized module
    """
    if issubclass(task_class, FinetuningTask):
        model_class = parse_model(model)
        model = model_class(config.IM_SHAPE, 1, 0.7)
        module = task_class(
            pretrained_model=model,
            im_shape=config.IM_SHAPE,
        )
    else:
        module = task_class(
            model_class=model,
            im_shape=config.IM_SHAPE,
        )
    return module


def get_module_dl(
    task_class: Type[FinetuningTask | LightningModule],
    model: str,
    batch_size: int = 3,
    num_samples: int = 3,
):
    """Returns module and dataloader ready for tests

    Args:
        task_class (Type[FinetuningTask  |  LightningModule]): Class of the task
        model (str): model name
        batch_size (int, optional): Batch size of dataloader. Defaults to 3.
        num_samples (int, optional): Samples in dataset. Defaults to 3.

    Returns:
        _type_: _description_
    """
    label = torch.tensor(1.0)
    if issubclass(task_class, MotionPretrainingTask):
        soft_util: ToSoftLabel = ToSoftLabel.motion_config()
        label = soft_util.value_to_softlabel(label)
    elif issubclass(task_class, SSIMPretrainingTask):
        soft_util: ToSoftLabel = ToSoftLabel.ssim_config()
        label = soft_util.value_to_softlabel(label)
    elif issubclass(task_class, (MRArtFinetuningTask, MRArtScratchTask)):
        label = torch.tensor(1)
    ds = [
        {"data": torch.randn(config.IM_SHAPE), "label": label, "identifier": "test"}
    ] * num_samples
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    module = parse_module(task_class, model)
    return module, dl
