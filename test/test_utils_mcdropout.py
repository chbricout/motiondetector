from typing import Callable, Type
import itertools
from comet_ml import testlib
from lightning import LightningModule
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
import pytest
import torch.utils
import torch.utils.data
from src.training.pretrain_logic import (
    BinaryPretrainingTask,
    MotionPretrainingTask,
    PretrainingTask,
    SSIMPretrainingTask,
)
from src.training.scratch_logic import AMPSCZScratchTask, MRArtScratchTask
from src.training.transfer_logic import (
    BaseFinalTrain,
    TransferTask,
    MrArtTransferTask,
    AMPSCZTransferTask,
)
from src.utils.mcdropout import (
    bincount2d,
    transfer_mcdropout,
    finetune_pred_to_df,
    predict_mcdropout,
    pretrain_mcdropout,
    pretrain_pred_to_df,
)
from src import config
from src.utils.test import get_module_dl


@pytest.mark.parametrize(
    "convert_func,has_bincount",
    [
        (pretrain_pred_to_df, False),
        (finetune_pred_to_df, True),
    ],
)
def test_pred_to_df(convert_func: Callable, has_bincount: bool):
    n_samples = 10
    n_preds = 100

    samples = torch.randint(0, 2, (n_samples, n_preds)).float()
    labels = torch.randint(0, 2, (n_samples,)).tolist()
    identifiers = ["test"] * n_samples
    df = convert_func(samples, labels, identifiers)
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "label" in df.columns
    assert "predictions" in df.columns
    assert "identifier" in df.columns

    assert not has_bincount or "count" in df.columns
    assert not has_bincount or "confidence" in df.columns
    assert not has_bincount or (df["confidence"] <= 0).sum() == 0
    assert not has_bincount or (df["confidence"] >= 1).sum() == 0
    assert not has_bincount or "max_classe" in df.columns

    assert len(df["predictions"].iloc[0]) == n_preds
    assert len(df) == n_samples


@pytest.mark.parametrize(
    "n_classes",
    [
        (2),
        (3),
        (3),
    ],
)
def test_bincount(n_classes: int):
    n_samples = 10
    n_indiv=20
    arr = torch.randint(0, n_classes, (n_indiv, n_samples))
    res = bincount2d(arr)

    assert res.shape == (n_indiv, n_classes)


@pytest.mark.parametrize(
    "task_class,model",
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
    ),
)
def test_predict_mc_dropout(
    task_class: Type[TransferTask | LightningModule], model: str
):
    n_preds = 2
    n_samples = 2

    module, dl = get_module_dl(task_class, model, n_samples, n_samples)

    preds, labels, identifiers = predict_mcdropout(
        module, dataloader=dl, n_preds=n_preds
    )
    assert preds.shape == (n_samples, n_preds)
    assert len(labels) == n_samples
    assert len(identifiers) == n_samples






@pytest.mark.parametrize(
    "task_class,model",
    itertools.product(
        [
            MotionPretrainingTask,
            SSIMPretrainingTask,
            BinaryPretrainingTask,
        ],
        ["CNN", "RES", "SFCN", "CONV5_FC3", "SERES", "VIT"],
    ),
)
def test_pretrain_mcdropout(task_class: Type[PretrainingTask], model: str):
    n_preds = 2
    n_samples = 2

    module, dl = get_module_dl(task_class, model, n_samples, n_samples)

    df = pretrain_mcdropout(
        pl_module=module,
        dataloader=dl,
        experiment=testlib.TestExperiment(),
        n_preds=n_preds,
    )
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "label" in df.columns
    assert "predictions" in df.columns
    assert "identifier" in df.columns

    assert len(df["predictions"].iloc[0]) == n_preds
    assert len(df) == n_samples


@pytest.mark.parametrize(
    "task_class,model",
    itertools.product(
        [
            MrArtTransferTask,
            MRArtScratchTask,
            AMPSCZTransferTask,
            AMPSCZScratchTask,
        ],
        ["CNN", "RES", "SFCN", "CONV5_FC3", "SERES", "VIT"],
    ),
)
def test_finetune_mcdropout(
    task_class: Type[TransferTask | BaseFinalTrain], model: str
):
    n_preds = 2
    n_samples = 2

    module, dl = get_module_dl(task_class, model, n_samples, n_samples)

    df = transfer_mcdropout(
        pl_module=module,
        dataloader=dl,
        experiment=testlib.TestExperiment(),
        n_preds=n_preds,
        log_figs=True,
    )
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "label" in df.columns
    assert "predictions" in df.columns
    assert "identifier" in df.columns

    assert "count" in df.columns
    assert "confidence" in df.columns
    assert (df["confidence"] <= 0).sum() == 0
    assert (df["confidence"] >= 1).sum() == 0
    assert "max_classe" in df.columns

    assert len(df["predictions"].iloc[0]) == n_preds
    assert len(df) == n_samples
