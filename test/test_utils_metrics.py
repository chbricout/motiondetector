from matplotlib.figure import Figure
import pytest
from src.utils.metrics import separation_capacity


@pytest.mark.parametrize(
    "label,pred,expect_acc",
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.8], 1),
        ([0, 1, 0, 1], [0.1, 0.2, 0.6, 0.8], 0.5),
        ([0, 1, 2], [0.1, 0.1, 0.1], 1 / 3),
        ([0, 0], [0.6, 0.8], 1),
    ],
)
def test_separation_capacity(label, pred, expect_acc):
    acc, fig, thresh = separation_capacity(label, pred)

    assert abs(acc - expect_acc) < 1e-4
    assert fig is not None
    assert type(fig) == Figure
    assert len(thresh) == max(label)
