import itertools
from typing import Type
import pytest

from src.training.eval import get_pred_from_pretrain
from src.training.lightning_logic import (
    BinaryPretrainingTask,
    MotionPretrainingTask,
    PretrainingTask,
    SSIMPretrainingTask,
)
from src.utils.test import get_module_dl


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
