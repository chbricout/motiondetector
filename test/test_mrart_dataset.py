import os
from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt, TestMrArt


class TestTrainDataset:
    ds = TrainMrArt.lab()

    def test_number(self):
        assert len(self.ds) == 258

    def test_structure(self):
        assert "data" in self.ds[0]
        assert "label" in self.ds[0]

    def test_existence(self):
        assert os.path.exists(self.ds[0]["data"])


class TestValDataset:
    ds = ValMrArt.lab()

    def test_number(self):
        assert len(self.ds) == 89

    def test_structure(self):
        assert "data" in self.ds[0]
        assert "label" in self.ds[0]

    def test_existence(self):
        assert os.path.exists(self.ds[0]["data"])


class TestTestDataset:
    ds = TestMrArt.lab()

    def test_number(self):
        assert len(self.ds) == 89

    def test_structure(self):
        assert "data" in self.ds[0]
        assert "label" in self.ds[0]

    def test_existence(self):
        assert os.path.exists(self.ds[0]["data"])
