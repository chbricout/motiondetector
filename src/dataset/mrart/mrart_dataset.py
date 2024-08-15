"""
Module to use the MR-ART dataset from python (require split csv files)
"""

from typing import Callable
from monai.data.dataset import Dataset, CacheDataset
from monai.data.dataloader import DataLoader
import pandas as pd
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import FinetuneTransform


class BaseMrArt(CacheDataset, BaseDataset):
    """
    Base dataset for common logic in MR-ART Data
    """

    csv_path: str

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.file = pd.read_csv(self.csv_path, index_col=0)
        self.file["identifier"] = self.file["data"].apply(BaseMrArt.extract_identifier)
        self.file["data"] = prefix + self.file["data"]

        super().__init__(
            self.file[["data", "identifier", "label"]].to_dict("records"), transform
        )

    @staticmethod
    def extract_identifier(path: str) -> str:
        """Retrieve volume identifier from path

        Args:
            path (str): path of file in dataframe

        Returns:
            str: formatted identifier
        """
        return "_".join(path.split("/")[2:4])


class TrainMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the train split of MR-ART (in finetune).
    It relies on the "train_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/train_preproc.csv"


class ValMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the validation split of MR-ART (in finetune).
    It relies on the "val_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/val_preproc.csv"


class TestMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the test split of MR-ART (in finetune).
    It relies on the "test_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/test_preproc.csv"


class MRArtDataModule(BaseDataModule):
    """
    Lightning data module to use MR-ART data in lightning trainers (for finetune)
    """

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)
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
