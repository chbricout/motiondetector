"""
Module to use the synthetic motion pretraining dataset from python (require split csv files)
"""

from typing import Callable
from torch.utils.data import Dataset
import pandas as pd
from monai.data.dataloader import DataLoader

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


class BasePretrain(Dataset, BaseDataset):
    """
    Base dataset for common logic in AMPSCZ Data
    """

    csv_path: str

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.file = pd.read_csv(self.csv_path, index_col=0)

        self.file["identifier"] = self.file["data"].apply(
            BasePretrain.extract_identifier
        )
        self.file["data"] = prefix + self.file["data"]
        self.transform = transform

        #Define default label
        self.define_label()

    @staticmethod
    def extract_identifier(path: str) -> str:
        """Retrieve volume identifier from path

        Args:
            path (str): path of file in dataframe

        Returns:
            str: formatted identifier
        """
        return path.split("/")[-1].split(".")[0]

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
        self.file["label"] = self.file[parse_label_from_task(task)]


class PretrainTrain(BasePretrain):
    """
    Pytorch Dataset to use the train split of synthetic motion dataset (in pretrain).
    It relies on the "train.csv" file
    """

    csv_path: str = "src/dataset/pretraining/train.csv"


class PretrainVal(BasePretrain):
    """
    Pytorch Dataset to use the validation split of synthetic motion dataset (in pretrain).
    It relies on the "val.csv" file
    """

    csv_path: str = "src/dataset/pretraining/val.csv"


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
            num_workers=9,
            prefetch_factor=2,
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
            num_workers=9,
            prefetch_factor=2,
        )
