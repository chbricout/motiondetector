import itertools
from typing import Type
from comet_ml.testlib import TestExperiment
from matplotlib.figure import Figure
import pytest
import torch

from src.training.eval import get_box_plot, get_correlations, get_pred_from_pretrain
from src.training.lightning_logic import (
    BinaryPretrainingTask,
    MotionPretrainingTask,
    PretrainingTask,
    SSIMPretrainingTask,
)
from src.utils.test import get_module_dl, parse_module


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
    n_samples = 2
    module, dl = get_module_dl(task_class, model, n_samples, n_samples)
    df = get_pred_from_pretrain(module, dl)

    assert len(df) == 2
    assert "label" in df.columns
    assert "identifier" in df.columns
    assert "pred" in df.columns


def test_get_box_plot():
    n_sample = 20
    predictions = torch.randn(n_sample).tolist()
    labels = torch.randint(0,3,(n_sample,)).tolist()
    fig = get_box_plot(predictions, labels)
    assert fig is not None
    assert isinstance(fig, Figure)

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
def test_get_correlations(task_class: Type[PretrainingTask], model: str):
    module = parse_module(task_class, model)
    exp = TestExperiment()
    get_correlations(module, exp)