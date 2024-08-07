"""
Module to use the synthetic motion pretraining dataset from python (require split csv files)
"""

from torch.utils.data import Dataset
import pandas as pd
from monai.data.dataloader import DataLoader

from src.config import BIN_RANGE, N_BINS
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import LoadSynth


class PretrainTrain(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of synthetic motion dataset (in pretrain).
    It relies on the "train.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/pretraining/train.csv", index_col=0)
        if prefix != "":
            self.files["data"] = self.files["data"].str.replace(
                "/home/cbricout/scratch/", prefix
            )
        self.files["label"] = self.files["motion_mm"]
        self.transform = transform

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
            self.files["data"] = self.files["data"].str.replace(
                "/home/cbricout/scratch/", prefix
            )
        self.files["label"] = self.files["motion_mm"]
        self.transform = transform

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

    def __init__(self, narval=True, batch_size: int = 32):
        super().__init__(narval, batch_size)
        self.load_tsf = LoadSynth(num_bins=N_BINS, bin_range=BIN_RANGE)
        self.val_ds_class = PretrainVal
        self.train_ds_class = PretrainTrain

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=20,
            prefetch_factor=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=10,
            prefetch_factor=2,
        )
