"""
Module to use the synthetic motion pretraining dataset from python (require split csv files)
"""

import os
from typing import Callable, Self
from torch.utils.data import Dataset
import pandas as pd
from monai.data.dataloader import DataLoader

from src import config
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import LoadSynth
from src.utils.task import label_from_task


class BasePretrain(Dataset, BaseDataset):
    """
    Base dataset for common logic in AMPSCZ Data
    """

    huge_path: str
    veryhuge_path: str

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.csv_path = self.veryhuge_path if config.IS_NARVAL else self.huge_path
        self.file = pd.read_csv(self.csv_path, index_col=0)

        self.file["identifier"] = self.file["identifier"]
        self.file["data"] = prefix + self.file["data"]
        self.transform = transform

        # Define default label
        self.define_label()

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        data = self.file.iloc[idx].to_dict()
        if self.transform:
            data = self.transform(data)
        return data

    def define_label(self, task: str = "MOTION"):
        """Setup dataset label corresponding to task

        Args:
            task (str, optional): pretraining task. Defaults to "MOTION".
        """
        if task=="BINARY":
            self.file["label"] = self.file[label_from_task(task)].astype(int)
        else:
            self.file["label"] = self.file[label_from_task(task)].astype(float)




class PretrainTrain(BasePretrain):
    """
    Pytorch Dataset to use the train split of synthetic motion dataset (in pretrain).
    It relies on the "train.csv" file
    """

    huge_path: str = "src/dataset/pretraining/huge-train.csv"
    veryhuge_path: str = "src/dataset/pretraining/veryhuge-train.csv"


class PretrainVal(BasePretrain):
    """
    Pytorch Dataset to use the validation split of synthetic motion dataset (in pretrain).
    It relies on the "val.csv" file
    """

    huge_path: str = "src/dataset/pretraining/veryhuge-val.csv"
    veryhuge_path: str = "src/dataset/pretraining/veryhuge-val.csv"


class PretrainTest(BasePretrain):
    """
    Pytorch Dataset to use the validation split of synthetic motion dataset (in pretrain).
    It relies on the "val.csv" file
    """

    huge_path: str = "src/dataset/pretraining/veryhuge-test.csv"
    veryhuge_path: str = "src/dataset/pretraining/veryhuge-test.csv"


class PretrainingDataModule(BaseDataModule):
    """
    Lightning data module to use synthetic motion pretraining data in lightning trainers
    """

    def __init__(self, batch_size: int = 32, task: str = "MOTION"):
        super().__init__(batch_size)
        self.load_tsf = LoadSynth.from_task(task)
        self.val_ds_class = PretrainVal
        self.train_ds_class = PretrainTrain
        self.task = task

    def train_dataloader(self):
        self.train_ds.define_label(self.task)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=9,
            prefetch_factor=2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self, num_workers=9):
        self.val_ds.define_label(self.task)
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2,
        )
