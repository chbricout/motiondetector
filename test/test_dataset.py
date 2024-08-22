import itertools
import os
import pytest
from src.dataset.ampscz.ampscz_dataset import (
    PretrainTrainAMPSCZ,
    PretrainValAMPSCZ,
    PretrainTestAMPSCZ,
    TransferTrainAMPSCZ,
    TransferValAMPSCZ,
    TransferTestAMPSCZ,
)
from src.dataset.hcpep.hcpep_dataset import TrainHCPEP, ValHCPEP, TestHCPEP
from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt, TestMrArt
from src.dataset.pretraining.pretraining_dataset import (
    PretrainTrain,
    PretrainVal,
    BasePretrain,
)
from src.dataset.base_dataset import BaseDataset


@pytest.mark.parametrize(
    "dataset,datapoints",
    [
        (PretrainTrainAMPSCZ, 254),
        (PretrainValAMPSCZ, 33),
        (PretrainTestAMPSCZ, 32),
        (TrainHCPEP, 115),
        (ValHCPEP, 14),
        (TestHCPEP, 14),
    ],
)
def test_no_label_datasets(dataset: BaseDataset, datapoints: int):
    ds = dataset.lab()
    assert len(ds) == datapoints
    assert "data" in ds[0]
    assert "identifier" in ds[0]
    assert os.path.exists(ds[0]["data"])


@pytest.mark.parametrize(
    "dataset,datapoints",
    [
        (TrainMrArt, 258),
        (ValMrArt, 89),
        (TestMrArt, 89),
        (TransferTrainAMPSCZ, 115),
        (TransferValAMPSCZ, 39),
        (TransferTestAMPSCZ, 42),
    ],
)
def test_labelled_datasets(dataset: BaseDataset, datapoints: int):
    ds = dataset.lab()
    assert len(ds) == datapoints
    assert "data" in ds[0]
    assert "identifier" in ds[0]
    assert "label" in ds[0]
    assert os.path.exists(ds[0]["data"])


@pytest.mark.parametrize(
    "dataset,datapoints, task",
    [
        (*conf[0], conf[1])
        for conf in itertools.product(
            [
                (PretrainTrain, 7380),
                (PretrainVal, 940),
            ],
            ["MOTION", "SSIM", "BINARY"],
        )
    ],
)
def test_generated_datasets(dataset: BasePretrain, datapoints: int, task: str):
    ds = dataset.lab()
    ds.define_label()
    assert len(ds) == datapoints
    assert "data" in ds[0]
    assert "identifier" in ds[0]
    assert "label" in ds[0]
    assert "motion_mm" in ds[0]
    assert "ssim_loss" in ds[0]
    assert "motion_binary" in ds[0]
    assert os.path.exists(ds[0]["data"])
