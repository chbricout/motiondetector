"""
Module to use the AMPSCZ dataset from python (require split csv files)
"""

from typing import Callable
from monai.data.dataset import Dataset, CacheDataset
from monai.data.dataloader import DataLoader
import pandas as pd
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import FinetuneTransform


class PretrainTrainAMPSCZ(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "train"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )


class PretrainValAMPSCZ(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the validation split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "val"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )


class PretrainTestAMPSCZ(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the test split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "test"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )


class FinetuneTrainAMPSCZ(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/finetune.csv", index_col=0)
        self.files = self.files[self.files["group"] == "train"]
        self.files["data"] = prefix + self.files["data"]
        self.files["label"] = self.files["motion"].astype(float)
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "label", "sub_id", "ses_id"]].to_dict("records"),
            transform,
        )


class FinetuneValAMPSCZ(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the validation split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/finetune.csv", index_col=0)
        self.files = self.files[self.files["group"] == "val"]
        self.files["data"] = prefix + self.files["data"]
        self.files["label"] = self.files["motion"].astype(float)
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "label", "sub_id", "ses_id"]].to_dict("records"),
            transform,
        )


class FinetuneTestAMPSCZ(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the test split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/finetune.csv", index_col=0)
        self.files = self.files[self.files["group"] == "test"]
        self.files["data"] = prefix + self.files["data"]
        self.files["label"] = self.files["motion"].astype(float)
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "label", "sub_id", "ses_id"]].to_dict("records"),
            transform,
        )


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
