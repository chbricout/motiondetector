"""
Module to use the MR-ART dataset from python (require split csv files)
"""

from typing import Callable
from monai.data.dataset import Dataset, CacheDataset
from monai.data.dataloader import DataLoader
import pandas as pd
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import FinetuneTransform


class TrainMrArt(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of MR-ART (in finetune).
    It relies on the "train_preproc.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/mrart/train_preproc.csv", index_col=0)
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)


class ValMrArt(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the validation split of MR-ART (in finetune).
    It relies on the "val_preproc.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/mrart/val_preproc.csv")
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)


class TestMrArt(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the test split of MR-ART (in finetune).
    It relies on the "test_preproc.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/mrart/test_preproc.csv")
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)


class MRArtDataModule(BaseDataModule):
    """
    Lightning data module to use MR-ART data in lightning trainers (for finetune)
    """

    def __init__(self, narval=True, batch_size: int = 32):
        super().__init__(narval, batch_size)
        self.load_tsf: Callable = FinetuneTransform()
        self.val_ds_class = ValMrArt
        self.train_ds_class = TrainMrArt

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=25,
            prefetch_factor=3,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=14,
            pin_memory=True,
            persistent_workers=True,
        )
