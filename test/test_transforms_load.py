from comet_ml import Model
import numpy as np
import pytest
import torch
from src import config
from src.network.cnn_net import CNNModel
from src.network.conv5_fc3_net import Conv5FC3Model
from src.network.res_net import ResModel
from src.network.seres_net import SEResModel
from src.network.sfcn_net import SFCNModel
from src.network.vit_net import ViTModel
from src.transforms.load import PretrainerTransform, ToSoftLabel
from src.config import (
    SSIM_BIN_RANGE,
    SSIM_BIN_STEP,
    SSIM_N_BINS,
    MOTION_BIN_RANGE,
    MOTION_BIN_STEP,
    MOTION_N_BINS,
)
from monai.data import Dataset


@pytest.mark.parametrize(
    "bin_range,bin_step,n_bins,value_range,expected_precision",
    [
        (SSIM_BIN_RANGE, SSIM_BIN_STEP, SSIM_N_BINS, (0, 0.4), 1e-5),
        (MOTION_BIN_RANGE, MOTION_BIN_STEP, MOTION_N_BINS, (0, 4), 1e-4),
    ],
)
def test_soft_label_reflexive(
    bin_range, bin_step, n_bins, value_range, expected_precision
):
    x = torch.arange(value_range[0], value_range[1] + bin_step, bin_step)

    to_soft = ToSoftLabel(
        keys="data", backup_keys="", bin_range=bin_range, bin_step=bin_step
    )

    soft_x = to_soft.value_to_softlabel(x)
    hard_x = to_soft.logsoft_to_hardlabel(soft_x.log())
    diff = x - hard_x
    assert (x - hard_x).abs().max() < expected_precision
    assert (x - hard_x).abs().sum() < expected_precision
    assert soft_x.shape == (len(x), n_bins)


@pytest.mark.parametrize(
    "val",
    torch.arange(
        0,
        1.04,
        0.2,
    ),
)
def test_soft_label_scalar(val: float):
    bin_range = (-0.95, 1.3)
    bin_step = 0.05
    n_bins = np.ceil((bin_range[1] - bin_range[0]) / bin_step)
    n_samples = 1
    x = val

    to_soft = ToSoftLabel(
        keys="data", backup_keys="", bin_range=bin_range, bin_step=bin_step
    )

    soft_x = to_soft.value_to_softlabel(x)
    hard_x = to_soft.logsoft_to_hardlabel(soft_x.log())

    assert (x - hard_x).abs().sum() < 1e-5
    assert soft_x.shape == (n_bins,)


@pytest.mark.parametrize(
    "model_to_test",
    [CNNModel, ResModel, Conv5FC3Model, SEResModel, SFCNModel, ViTModel],
)
def test_pretrain_transform(model_to_test: Model):
    net = model_to_test(config.IM_SHAPE, 10, 0.5)
    transf = PretrainerTransform("data", net.encoder)
    ds = Dataset(
        [
            {
                "data": torch.randn(config.IM_SHAPE),
            }
        ],
        transform=transf,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
