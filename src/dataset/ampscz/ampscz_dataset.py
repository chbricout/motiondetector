"""
Module to use the AMPSCZ dataset from python (require split csv files)
"""

from typing import Callable
from monai.data.dataset import Dataset, CacheDataset
from monai.data.dataloader import DataLoader
import pandas as pd
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import FinetuneTransform


class BaseAMPSCZ(CacheDataset, BaseDataset):
    """
    Base dataset for common logic in AMPSCZ Data
    """

    csv_path: str
    group: str
    labelled: bool

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.file = pd.read_csv(self.csv_path, index_col=0)
        self.file = self.file[self.file["group"] == self.group]
        self.file["data"] = prefix + self.file["data"]
        self.file["identifier"] = (
            self.file["sub_id_gs"] + "_" + self.file["ses_id_gs"].astype(str)
        )
        subset : pd.DataFrame = self.file
        if self.labelled:
            self.file["label"] = self.file["motion"].astype(float)
            subset = self.file[["data", "identifier", "label"]]
            
        else:
            subset = self.file[["data", "identifier"]]

        super().__init__(
                subset.to_dict("records"), transform
            )


class PretrainTrainAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the train split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "train"
    labelled: bool = False


class PretrainValAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the validation split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "val"
    labelled: bool = False


class PretrainTestAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the test split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "test"
    labelled: bool = False


class FinetuneTrainAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the train split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "train"
    labelled: bool = True


class FinetuneValAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the validation split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "val"
    labelled: bool = True


class FinetuneTestAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the test split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "test"
    labelled: bool = True


class AMPSCZDataModule(BaseDataModule):
    """
    Lightning data module to use AMPSCZ finetune data in lightning trainers
    """

    def __init__(self, narval: bool = True, batch_size: int = 32):
        super().__init__(narval, batch_size)
        self.load_tsf: Callable = FinetuneTransform()
        self.val_ds_class = FinetuneValAMPSCZ
        self.train_ds_class = FinetuneTrainAMPSCZ

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
