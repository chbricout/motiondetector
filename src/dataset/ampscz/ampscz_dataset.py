"""
Module to use the AMPSCZ dataset from python (require split csv files)
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch
import tqdm
from comet_ml import Model
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from torch.utils.data import ConcatDataset, Dataset

from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.transforms.load import FinetuneTransform


class BaseAMPSCZ(CacheDataset, BaseDataset):
    """
    Base dataset for common logic in AMPSCZ Data
    """

    csv_path: str
    group: str
    labelled: bool

    def __init__(
        self,
        transform: Callable | None = None,
        prefix: str = "",
        pretrained_model: Model | None = None,
    ):
        self.pretrained_model = pretrained_model
        self.file = pd.read_csv(self.csv_path, index_col=0)
        self.file = self.file[self.file["group"] == self.group]
        self.file["data"] = prefix + self.file["data"]
        self.file["identifier"] = (
            self.file["sub_id_gs"] + "_" + self.file["ses_id_gs"].astype(str)
        )
        subset: pd.DataFrame = self.file
        if self.labelled:
            self.file["label"] = self.file["score"].astype(int) - 2
            subset = self.file[["data", "identifier", "label"]]

        else:
            subset = self.file[["data", "identifier"]]

        super().__init__(subset.to_dict("records"), transform)

    @classmethod
    def get_weight(cls):
        file = pd.read_csv(cls.csv_path, index_col=0)
        file = file[file["group"] == cls.group]
        label_frequency = (
            file["score"].value_counts(normalize=False).sort_index().to_numpy()
        )
        tot_sample = len(file)
        label_weight = np.divide(tot_sample, 1 * label_frequency)
        print(label_weight)
        return torch.Tensor(label_weight.tolist())

    def setup(self, stage: str):
        self.val_ds = self.val_ds_class.from_env(self.load_tsf)
        self.train_ds = self.train_ds_class.from_env(self.load_tsf)
        if self.pretrained_model:
            self.val_ds = self.get_embeddings(self.val_ds)
            self.train_ds = self.get_embeddings(self.train_ds)
        logging.info(
            "Train dataset contains %d datas  \nVal dataset contains %d",
            len(self.train_ds),
            len(self.val_ds),
        )

    def get_embeddings(self, dataset: CacheDataset):
        cache_ds = []
        self.pretrained_model.cuda()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataset):
                with torch.autocast(device_type="cuda"):
                    if isinstance(batch["data"], torch.IntTensor):
                        batch["data"] = batch["data"].as_tensor()
                    batch["data"] = batch["data"].unsqueeze(0).cuda()

                    batch["data"] = (
                        self.pretrained_model(batch["data"]).cpu().squeeze(0)
                    )
                    cache_ds.append(batch)
        return cache_ds


class PretrainTrainAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the train split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "train"
    labelled: bool = False


class PretrainValAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the validation split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "val"
    labelled: bool = False


class PretrainTestAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the test split of the pretrain dedicated part of AMPSCZ
    It relies on the "pretrain.csv" file
    """

    csv_path: str = "src/dataset/ampscz/pretrain.csv"
    group: str = "test"
    labelled: bool = False


class TransferTrainAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the train split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "train"
    labelled: bool = True


class TransferValAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the validation split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "val"
    labelled: bool = True


class TransferTestAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the test split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/finetune.csv"
    group: str = "test"
    labelled: bool = True


class TransferExtraAMPSCZ(BaseAMPSCZ):
    """
    Pytorch Dataset to use the test split of the finetune dedicated part of AMPSCZ
    It relies on the "finetune.csv" file
    """

    csv_path: str = "src/dataset/ampscz/new_volume_for_test.csv"
    group: str = "test"
    labelled: bool = True


class FullTestAMPSCZ(BaseDataset, Dataset):
    def __init__(
        self,
        transform: Callable | None = None,
        prefix: str = "",
        pretrained_model: Model | None = None,
    ):
        self.ds = ConcatDataset(
            [
                TransferTestAMPSCZ(
                    transform=transform,
                    prefix=prefix,
                    pretrained_model=pretrained_model,
                ),
                TransferExtraAMPSCZ(
                    transform=transform,
                    prefix=prefix,
                    pretrained_model=pretrained_model,
                ),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        return self.ds[idx]


class AMPSCZDataModule(BaseDataModule):
    """
    Lightning data module to use AMPSCZ finetune data in lightning trainers
    """

    def __init__(self, batch_size: int = 32, pretrained_model: Model | None = None):
        super().__init__(batch_size)
        self.load_tsf: Callable = FinetuneTransform()
        self.val_ds_class = TransferValAMPSCZ
        self.train_ds_class = TransferTrainAMPSCZ
        self.pretrained_model = pretrained_model

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

    def setup(self, stage: str):
        self.val_ds = self.val_ds_class.from_env(self.load_tsf)
        self.train_ds = self.train_ds_class.from_env(self.load_tsf)
        if self.pretrained_model:
            self.val_ds = self.get_embeddings(self.val_ds)
            self.train_ds = self.get_embeddings(self.train_ds)
        logging.info(
            "Train dataset contains %d datas  \nVal dataset contains %d",
            len(self.train_ds),
            len(self.val_ds),
        )

    def get_embeddings(self, dataset: CacheDataset):
        cache_ds = []
        self.pretrained_model.cuda()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataset):
                # with torch.autocast(device_type="cuda"):
                if isinstance(batch["data"], torch.IntTensor):
                    batch["data"] = batch["data"].as_tensor()
                batch["data"] = batch["data"].unsqueeze(0).cuda()

                batch["data"] = self.pretrained_model(batch["data"]).cpu().squeeze(0)
                cache_ds.append(batch)
        return cache_ds
