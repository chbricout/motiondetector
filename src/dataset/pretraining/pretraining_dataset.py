import logging
from torch.utils.data import Dataset
import lightning as L
import pandas as pd
from monai.data.dataloader import DataLoader

from src.config import BIN_RANGE, N_BINS
from src.transforms.load import LoadSynth


class PretrainTrain(Dataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/pretraining/train.csv", index_col=0)
        self.files["label"] = self.files["motion_mm"]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files.iloc[idx].to_dict()
        if self.transform:
            data = self.transform(data)

        return data

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "")


class PretrainVal(Dataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/pretraining/val.csv", index_col=0)
        self.files["label"] = self.files["motion_mm"]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files.iloc[idx].to_dict()
        if self.transform:
            data = self.transform(data)

        return data

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "")


class PretrainingDataModule(L.LightningDataModule):
    def __init__(self, narval=True, batch_size: int = 32):
        super().__init__()
        self.narval = narval
        self.batch_size = batch_size
        self.load_tsf = LoadSynth(num_bins=N_BINS, bin_range=BIN_RANGE)

    def setup(self, stage: str):
        self.val_ds = (
            PretrainVal.narval(self.load_tsf)
            if self.narval
            else PretrainVal.lab(self.load_tsf)
        )
        self.train_ds = (
            PretrainTrain.narval(self.load_tsf)
            if self.narval
            else PretrainTrain.lab(self.load_tsf)
        )
        logging.info(
            f"Train dataset contains {len(self.train_ds)} datas  \nVal dataset contains {len(self.val_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=13,
            prefetch_factor=4,
            drop_last =True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=5,
            prefetch_factor=2,
        )
