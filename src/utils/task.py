"""Module to define any utility function to use tasks"""

import sys
from os import path
from typing import Type

import torch
from lightning import Trainer

from src import config
from src.training.pretrain_logic import MotionPretrainingTask


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


def load_pretrain_from_ckpt(ckpt_path: str):
    model_str = parse_transfer_path(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    checkpoint["state_dict"].pop(
        "label_loss.pos_weight", None
    )  # Safely removes the key
    module = MotionPretrainingTask(model_str, config.IM_SHAPE)
    module.load_state_dict(checkpoint["state_dict"])
    return module, model_str


def parse_transfer_path(ckpt_path: str):
    base_name = path.basename(ckpt_path)
    model_str, *_ = base_name.split("-")
    return model_str
