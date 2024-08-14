import itertools
from typing import Type
from lightning import LightningModule
import pytest
import torch
from src import config
from src.training.lightning_logic import (
    BaseTrain,
    FinetuningTask,
    MotionPretrainingTask,
    PretrainingTask,
    SSIMPretrainingTask,
    BinaryPretrainingTask,
    MRArtFinetuningTask,
    MRArtScratchTask,
    AMPSCZFinetuningTask,
    AMPSCZScratchTask,
)
from src.utils.test import get_module_dl


@pytest.mark.parametrize(
    "task_class,model",
    list(
        itertools.product(
            [
                MotionPretrainingTask,
                SSIMPretrainingTask,
                BinaryPretrainingTask,
                MRArtFinetuningTask,
                MRArtScratchTask,
                AMPSCZFinetuningTask,
                AMPSCZScratchTask,
            ],
            ["CNN", "RES", "SFCN", "CONV5_FC3", "SERES", "VIT"],
        )
    ),
)
class TestLightningModule:
    def test_train_step(
        self, task_class: Type[FinetuningTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 3, 3)

        train_loss = module.training_step(next(iter(dl)), 1)
        assert train_loss is not None
        assert train_loss.numel() == 1
        assert type(train_loss.item()) is float

    def test_val_step(
        self, task_class: Type[FinetuningTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 3, 3)

        with torch.no_grad():
            val_loss = module.validation_step(next(iter(dl)), 1)
        assert val_loss is not None
        assert val_loss.numel() == 1
        assert type(val_loss.item()) is float

    def test_pred_step(
        self, task_class: Type[FinetuningTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 3, 3)

        with torch.no_grad():
            predict = module.predict_step(next(iter(dl)), 1)
        assert predict is not None
        assert predict.numel() == 3
        if issubclass(task_class, (MotionPretrainingTask, SSIMPretrainingTask)):
            assert predict.is_floating_point()
        else:
            assert not predict.is_floating_point()
