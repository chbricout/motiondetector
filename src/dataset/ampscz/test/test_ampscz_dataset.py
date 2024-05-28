import os
from src.dataset.ampscz.ampscz_dataset import (
    PretrainTrainAMPSCZ,
    PretrainValAMPSCZ,
    PretrainTestAMPSCZ,
    FinetuneTrainAMPSCZ,
    FinetuneValAMPSCZ,
    FinetuneTestAMPSCZ,
)


class TestPretrainTrainDataset:
    pretrain_ds = PretrainTrainAMPSCZ.lab()

    def test_number(self):
        assert len(self.pretrain_ds) == 254

    def test_structure(self):
        assert "data" in self.pretrain_ds[0]

    def test_existence(self):
        assert os.path.exists(self.pretrain_ds[0]["data"])


class TestPretrainValDataset:
    pretrain_ds = PretrainValAMPSCZ.lab()

    def test_number(self):
        assert len(self.pretrain_ds) == 33

    def test_structure(self):
        assert "data" in self.pretrain_ds[0]

    def test_existence(self):
        assert os.path.exists(self.pretrain_ds[0]["data"])


class TestPretrainTestDataset:
    pretrain_ds = PretrainTestAMPSCZ.lab()

    def test_number(self):
        assert len(self.pretrain_ds) == 32

    def test_structure(self):
        assert "data" in self.pretrain_ds[0]
     
        

    def test_existence(self):
        assert os.path.exists(self.pretrain_ds[0]["data"])


class TestFinetuneTrainDataset:
    finetune_ds = FinetuneTrainAMPSCZ.lab()

    def test_number(self):
        assert len(self.finetune_ds) == 115
        assert "label" in self.finetune_ds[0]

    def test_structure(self):
        assert "data" in self.finetune_ds[0]

    def test_existence(self):
        assert os.path.exists(self.finetune_ds[0]["data"])


class TestFinetuneValDataset:
    finetune_ds = FinetuneValAMPSCZ.lab()

    def test_number(self):
        assert len(self.finetune_ds) == 39

    def test_structure(self):
        assert "data" in self.finetune_ds[0]
        assert "label" in self.finetune_ds[0]


    def test_existence(self):
        assert os.path.exists(self.finetune_ds[0]["data"])


class TestFinetuneTestDataset:
    finetune_ds = FinetuneTestAMPSCZ.lab()

    def test_number(self):
        assert len(self.finetune_ds) == 42

    def test_structure(self):
        assert "data" in self.finetune_ds[0]
        assert "label" in self.finetune_ds[0]
        

    def test_existence(self):
        assert os.path.exists(self.finetune_ds[0]["data"])
