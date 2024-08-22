"""Module to provide utility function for test purposes only !"""

from typing import Type

from lightning import LightningModule
import torch
from src import config
from src.network.utils import parse_model
from src.training.pretrain_logic import MotionPretrainingTask, SSIMPretrainingTask
from src.training.scratch_logic import MRArtScratchTask
from src.training.transfer_logic import (
    TransferTask,
    MrArtTransferTask,
)
from src.transforms.load import ToSoftLabel


def parse_module(
    task_class: Type[TransferTask | LightningModule], model: str
) -> LightningModule:
    """Return correct lightning module with default init depending on the type of task
    for test purposes

    Args:
        task_class (Type[FinetuningTask  |  LightningModule]): Class of the task
        model (str): model name

    Returns:
        LightningModule: initialized module
    """
    if issubclass(task_class, TransferTask):
        module = task_class(
            input_size=10,
        )
    else:
        module = task_class(
            model_class=model,
            im_shape=config.IM_SHAPE,
        )
    return module


def get_dl(
    task_class: Type[TransferTask | LightningModule],
    batch_size: int = 2,
    num_samples: int = 2,
):
    """Create a minimal dataloader to test functionalities

    Args:
        task_class (Type[TransferTask  |  LightningModule]):
            Task intended for the dataloader labels
        batch_size (int, optional): batch size. Defaults to 2.
        num_samples (int, optional): number of samples. Defaults to 2.

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
    elif issubclass(task_class, (MrArtTransferTask, MRArtScratchTask)):
        label = torch.tensor(1)

    data = torch.randn(config.IM_SHAPE)
    if issubclass(task_class, TransferTask):
        data = torch.randn(10)

    ds = [
        {
            "data": data,
            "label": label,
            "identifier": "test",
            "motion_mm": torch.tensor(1.0),
            "ssim_loss": torch.tensor(1.0),
            "motion_binary": True,
        }
    ] * num_samples
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    return dl


def get_module_dl(
    task_class: Type[TransferTask | LightningModule],
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

    module = parse_module(task_class, model)
    dl = get_dl(task_class=task_class, batch_size=batch_size, num_samples=num_samples)
    return module, dl
