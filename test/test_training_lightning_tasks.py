import itertools
from typing import Type
from lightning import LightningModule
import pytest
import torch
from src.training.pretrain_logic import (
    BinaryPretrainingTask,
    MotionPretrainingTask,
    SSIMPretrainingTask,
)
from src.training.scratch_logic import AMPSCZScratchTask, MRArtScratchTask
from src.training.transfer_logic import (
    TransferTask,
    MrArtTransferTask,
    AMPSCZTransferTask,
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
                MrArtTransferTask,
                MRArtScratchTask,
                AMPSCZTransferTask,
                AMPSCZScratchTask,
            ],
            ["CNN", "RES", "SFCN", "CONV5_FC3", "SERES", "VIT"],
        )
    ),
)
class TestLightningModule:
    def test_train_step(
        self, task_class: Type[TransferTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 2, 2)

        train_loss = module.training_step(next(iter(dl)), 1)
        assert train_loss is not None
        assert train_loss.numel() == 1
        assert type(train_loss.item()) is float

    def test_val_step(
        self, task_class: Type[TransferTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 2, 2)

        with torch.no_grad():
            val_loss = module.validation_step(next(iter(dl)), 1)
        assert val_loss is not None
        assert val_loss.numel() == 1
        assert type(val_loss.item()) is float

    def test_pred_step(
        self, task_class: Type[TransferTask | LightningModule], model: str
    ):
        module, dl = get_module_dl(task_class, model, 2, 2)

        with torch.no_grad():
            predict = module.predict_step(next(iter(dl)), 1)
        assert predict is not None
        assert predict.numel() == 2
        if issubclass(task_class, (MotionPretrainingTask, SSIMPretrainingTask)):
            assert predict.is_floating_point()
        else:
            assert not predict.is_floating_point()
