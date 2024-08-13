from typing import Callable
import itertools
from lightning import LightningModule
import torch
import pytest
import torch.utils
import torch.utils.data
from src.training.lightning_logic import (
    MotionPretrainingTask,
    SSIMPretrainingTask,
    BinaryPretrainingTask,
    MRArtFinetuningTask,
    MRArtScratchTask,
    AMPSCZFinetuningTask,
    AMPSCZScratchTask,
)
from src.utils.mcdropout import (
    bincount2d,
    finetune_pred_to_df,
    predict_mcdropout,
    pretrain_pred_to_df,
)
from src import config


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
    assert len(df["predictions"].iloc[0]) == n_preds
    assert len(df) == n_samples


@pytest.mark.parametrize(
    "n_classes,indicate_bins",
    [
        (2, False),
        (3, True),
        (3, False),
    ],
)
def test_bincount(n_classes: int, indicate_bins: bool):
    n_samples = 100
    arr = torch.randint(0, n_classes, (n_samples,))
    if indicate_bins:
        res = bincount2d(arr, n_classes)
    else:
        res = bincount2d(arr)

    assert res.shape == (n_samples, n_classes)


@pytest.mark.parametrize(
    "task_class,model",
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
    ),
)
def test_predict_mc_dropout(task_class: LightningModule, model: str):
    n_preds = 2
    n_samples = 3
    ds = [{"data": torch.randn(config.IM_SHAPE), "label": 1, "identifier": "test"}] * n_samples
    dl = torch.utils.data.DataLoader(ds, batch_size=n_samples)

    module = task_class(
        model_class=model,
        im_shape=config.IM_SHAPE,
    )

    preds, labels, identifiers = predict_mcdropout(module, dataloader=dl, n_preds=n_preds)
    assert preds.shape == (n_samples, n_preds)
    assert len(labels) == n_samples
    assert len(identifiers) == n_preds