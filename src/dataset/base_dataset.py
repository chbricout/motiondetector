"""
Module containing base structure and functions for every dataset
"""

import abc
import logging
import re
from typing import Callable, Self
import lightning as L


def extract_sub(path: str) -> str:
    """Use regex to extract subject ID from the AMPSCZ path

    Args:
        path (str): path to AMPSCZ file (need to include the sub_id)

    Returns:
        str: the file sub_id
    """
    match_re = ".*(sub-[0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    sub_id = ""
    if match_res:
        sub_id = match_res.group(1)
    else:
        logging.error("Unable to identify subject id in path %s", path)

    return sub_id


class BaseDataset(abc.ABC):
    """
    Base dataset with class method to setup on lab computer or narval clusters
    """

    @classmethod
    def lab(cls, transform: Callable | None = None) -> Self:
        """Create a dataset with Neuro-iX laboratory computer settings

        Args:
            transform (Callable | None, optional):
                Transform to use on the dataset. Defaults to None.

        Returns:
            Self: Dataset on lab computer
        """
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform: Callable | None = None) -> Self:
        """Create a dataset with Neuro-iX laboratory computer settings

        Args:
            transform (Callable | None, optional):
                Transform to use on the dataset. Defaults to None.

        Returns:
            Self: Dataset on narval cluster
        """
        return cls(transform, "/home/cbricout/scratch/")


class BaseDataModule(abc.ABC, L.LightningDataModule):
    """
    Base lightning data module
    """

    load_tsf = None
    val_ds = None
    train_ds = None
    val_ds_class: BaseDataset
    train_ds_class: BaseDataset

    def __init__(self, narval: bool = True, batch_size: int = 32):
        super().__init__()
        self.narval = narval
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.val_ds = (
            self.val_ds_class.narval(self.load_tsf)
            if self.narval
            else self.val_ds_class.lab(self.load_tsf)
        )
        self.train_ds = (
            self.train_ds_class.narval(self.load_tsf)
            if self.narval
            else self.train_ds_class.lab(self.load_tsf)
        )
        logging.info(
            "Train dataset contains %d datas  \nVal dataset contains %d",
            len(self.train_ds),
            len(self.val_ds),
        )
