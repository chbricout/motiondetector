"""
Module to use the HCPEP dataset from python (require split csv files)
"""

from typing import Callable
from monai.data.dataset import  CacheDataset
import pandas as pd

from src.dataset.base_dataset import BaseDataset


class BaseHCPEP(CacheDataset, BaseDataset):
    """
    Base dataset for common logic in HCPEP Data
    """

    group: str

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.file = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.file = self.file[self.file["group"] == self.group]
        self.file["data"] = prefix + self.file["data"]
        self.file["identifier"] = self.file["sub_id"].astype(str) + "_" + self.file["ses_id"].astype(str)

        super().__init__(
            self.file[["data", "identifier"]].to_dict("records"), transform
        )


class TrainHCPEP(BaseHCPEP):
    """
    Pytorch Dataset to use the train split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    group = "train"


class ValHCPEP(BaseHCPEP):
    """
    Pytorch Dataset to use the validation split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    group = "val"


class TestHCPEP(BaseHCPEP):
    """
    Pytorch Dataset to use the test split of HCPEP (in pretrain synthetic dataset).
    It relies on the "pretrain.csv" file
    """

    group = "test"
