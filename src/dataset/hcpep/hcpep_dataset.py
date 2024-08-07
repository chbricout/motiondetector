"""
Module to use the HCPEP dataset from python (require split csv files)
"""

from monai.data.dataset import Dataset, CacheDataset
import pandas as pd

from src.dataset.base_dataset import BaseDataset


class TrainHCPEP(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the train split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "train"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )


class ValHCPEP(CacheDataset, BaseDataset):
    """
    Pytorch Dataset to use the validation split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "val"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )


class TestHCPEP(Dataset, BaseDataset):
    """
    Pytorch Dataset to use the test split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files["group"] == "test"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(
            self.files[["data", "sub_id", "ses_id"]].to_dict("records"), transform
        )
