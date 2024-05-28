from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt, TestMrArt
from src.transforms.base_transform import Preprocess, Augment
from torch import Tensor
import torch


class TestTransform:
    ds = TrainMrArt.lab()
    preprocess = Preprocess((160, 192, 160))
    augment = Augment()

    def test_preprocess(self):
        data = self.ds[0]
        res = self.preprocess(data)
        assert type(res) == dict
        assert isinstance(res["data"], Tensor)
        assert res["data"].shape == (1, 160, 192, 160)
        assert type(res["label"]) == int
        assert torch.isnan(res["data"]).any().logical_not()

    def test_augment(self):
        data = self.ds[0]
        res = self.preprocess(data)
        res = self.augment(res)
        assert type(res) == dict
        assert isinstance(res["data"], Tensor)
        assert res["data"].shape == (1, 160, 192, 160)
        assert type(res["label"]) == int
        assert torch.isnan(res["data"]).any().logical_not()
