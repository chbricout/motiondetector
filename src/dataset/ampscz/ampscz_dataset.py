import logging
import re
from monai.data.dataset import Dataset, CacheDataset
from monai.data.dataloader import DataLoader
from monai.transforms import Compose
import pandas as pd
import lightning as L
from src.transforms.load import FinetuneTransform


def extract_sub(path: str):
    match_re = ".*(sub-[0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return ""


class PretrainTrainAMPSCZ(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "train"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class PretrainValAMPSCZ(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "val"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class PretrainTestAMPSCZ(Dataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/ampscz/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "test"]
        self.files["data"] = prefix + self.files["data"]
        self.files = self.files.rename(
            columns={"sub_id_gs": "sub_id", "ses_id_gs": "ses_id"}
        )

        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class FinetuneTrainAMPSCZ(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
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

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class FinetuneValAMPSCZ(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
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

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class FinetuneTestAMPSCZ(Dataset):
    def __init__(self, transform=None, prefix: str = ""):
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

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class AMPSCZDataModule(L.LightningDataModule):
    def __init__(self, narval=True, batch_size: int = 32):
        super().__init__()
        self.narval = narval
        self.batch_size = batch_size
        self.load_tsf = FinetuneTransform()

    def setup(self, stage: str):
        self.val_ds = (
            FinetuneValAMPSCZ.narval(self.load_tsf)
            if self.narval
            else FinetuneValAMPSCZ.lab(self.load_tsf)
        )
        self.train_ds = (
            FinetuneTrainAMPSCZ.narval(self.load_tsf)
            if self.narval
            else FinetuneTrainAMPSCZ.lab(self.load_tsf)
        )
        logging.info(
            f"Train dataset contains {len(self.train_ds)} datas  \nVal dataset contains {len(self.val_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=25,
            pin_memory=True,
            prefetch_factor=3,
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
