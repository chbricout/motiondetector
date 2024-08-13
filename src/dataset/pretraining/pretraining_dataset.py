"""
Module to use the synthetic motion pretraining dataset from python (require split csv files)
"""

from torch.utils.data import Dataset
import pandas as pd
from monai.data.dataloader import DataLoader

from src.config import MOTION_BIN_RANGE, MOTION_N_BINS
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import LoadSynth


def parse_label_from_task(task: str) -> str:
    """Retrieve label column name in dataframe from Pretrain task

    Args:
        task (str): Task to pretrain on

    Returns:
        str: label for dataset
    """
    label = ""
    if task == "MOTION":
        label = "motion_mm"
    elif task == "SSIM":
        label = "ssim_loss"
    elif task == "BINARY":
        label = "motion_binary"
    return label


class PretrainTrain(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of synthetic motion dataset (in pretrain).
    It relies on the "train.csv" file
    """

    def __init__(self, transform=None, prefix: str = "", task: str = "MOTION"):
        self.files = pd.read_csv("src/dataset/pretraining/train.csv", index_col=0)
        if prefix != "":
            self.files["data"] = prefix + self.files["data"]
        self.transform = transform

    def define_label(self, task: str = "MOTION"):
        """Setup dataset label corresponding to task

        Args:
            task (str, optional): pretraining task. Defaults to "MOTION".
        """
        self.files["label"] = self.files[parse_label_from_task(task)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files.iloc[idx].to_dict()
        if self.transform:
            data = self.transform(data)

        return data


class PretrainVal(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the validation split of synthetic motion dataset (in pretrain).
    It relies on the "val.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/pretraining/val.csv", index_col=0)
        if prefix != "":
            self.files["data"] = prefix + self.files["data"]
        self.transform = transform

    def define_label(self, task: str = "MOTION"):
        """Setup dataset label corresponding to task

        Args:
            task (str, optional): pretraining task. Defaults to "MOTION".
        """
        self.files["label"] = self.files[parse_label_from_task(task)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files.iloc[idx].to_dict()
        if self.transform:
            data = self.transform(data)

        return data


class PretrainingDataModule(BaseDataModule):
    """
    Lightning data module to use synthetic motion pretraining data in lightning trainers
    """

    def __init__(self, narval=True, batch_size: int = 32, task: str = "MOTION"):
        super().__init__(narval, batch_size)
        self.load_tsf = LoadSynth.from_task(task)
        self.val_ds_class = PretrainVal
        self.train_ds_class = PretrainTrain
        self.task = task

    def train_dataloader(self):
        self.train_ds.define_label(self.task)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=12,
            prefetch_factor=6,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_ds.define_label(self.task)
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=4,
        )
