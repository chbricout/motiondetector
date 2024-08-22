from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from src.utils.metrics import separation_capacity


@pytest.mark.parametrize(
    "label,pred,expect_acc",
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.8], 1),
        ([0, 1, 0, 1], [0.1, 0.2, 0.6, 0.8], 0.75),
        ([0, 1, 2], [0.1, 0.1, 0.1], 1 / 3),
        ([0, 1, 2], [0.1, 0.2, 0.1], 2 / 3),
        ([0, 0], [0.6, 0.8], 1),
    ],
)
def test_separation_capacity(label, pred, expect_acc):

    arr = np.array([label, pred]).T
    df_val = pd.DataFrame(arr, columns=["label", "pred"])
    df_val["mode"] = "val"
    df_train = pd.DataFrame(arr, columns=["label", "pred"])
    df_train["mode"] = "train"

    acc, fig, thresh = separation_capacity(pd.concat([df_val, df_train]))

    assert abs(acc - expect_acc) < 1e-4
    assert fig is not None
    assert type(fig) == Figure
    assert len(thresh) <= max(max(label), 1)
