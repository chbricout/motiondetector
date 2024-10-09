"""
Module to use the MR-ART dataset from python (require split csv files)
"""

import logging
from typing import Callable
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader
import pandas as pd
import torch
import tqdm
from src.dataset.base_dataset import BaseDataModule, BaseDataset
from src.network.archi import Model
from src.transforms.load import FinetuneTransform, TransferTransform


class BaseMrArt(CacheDataset, BaseDataset):
    """
    Base dataset for common logic in MR-ART Data
    """

    csv_path: str

    def __init__(self, transform: Callable | None = None, prefix: str = ""):
        self.file = pd.read_csv(self.csv_path, index_col=0)
        self.file["identifier"] = self.file["data"].apply(BaseMrArt.extract_identifier)
        self.file["data"] = prefix + self.file["data"]

        super().__init__(
            self.file[["data", "identifier", "label"]].to_dict("records"), transform
        )

    @staticmethod
    def extract_identifier(path: str) -> str:
        """Retrieve volume identifier from path

        Args:
            path (str): path of file in dataframe

        Returns:
            str: formatted identifier
        """
        return "_".join(path.split("/")[2:4])


class TrainMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the train split of MR-ART (in finetune).
    It relies on the "train_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/train_preproc.csv"


class ValMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the validation split of MR-ART (in finetune).
    It relies on the "val_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/val_preproc.csv"


class TestMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the test split of MR-ART (in finetune).
    It relies on the "test_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/test_preproc.csv"


class MRArtDataModule(BaseDataModule):
    """
    Lightning data module to use MR-ART data in lightning trainers (for finetune)
    """

    def __init__(self, batch_size: int = 32, pretrained_model: Model | None = None):
        super().__init__(batch_size)

        self.load_tsf: Callable = FinetuneTransform()
        self.val_ds_class = ValMrArt
        self.train_ds_class = TrainMrArt
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
                with torch.autocast(device_type="cuda"):
                    if isinstance(batch["data"], torch.IntTensor):
                        batch["data"] = batch["data"].as_tensor()
                    batch["data"] = batch["data"].unsqueeze(0).cuda()

                    batch["data"] = (
                        self.pretrained_model(batch["data"]).cpu().squeeze(0)
                    )
                    cache_ds.append(batch)
        logging.error(f"Output shape {cache_ds[0]['data'].shape}")
        return cache_ds



class TrainUnbalancedMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the Unbalanced train split of MR-ART (in finetune).
    It relies on the "unbalanced_train_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/unbalanced_train_preproc.csv"


class ValUnbalancedMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the Unbalanced validation split of MR-ART (in finetune).
    It relies on the "unbalanced_val_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/unbalanced_val_preproc.csv"


class TestUnbalancedMrArt(BaseMrArt):
    """
    Pytorch Dataset to use the Unbalanced test split of MR-ART (in finetune).
    It relies on the "unbalanced_test_preproc.csv" file
    """

    csv_path = "src/dataset/mrart/unbalanced_test_preproc.csv"


class UnbalancedMRArtDataModule(MRArtDataModule):
    """
    Lightning data module to use Unbalanced MR-ART data in lightning trainers (for finetune)
    """

    def __init__(self, batch_size: int = 32, pretrained_model: Model | None = None):
        super().__init__(batch_size)

        self.load_tsf: Callable = FinetuneTransform()
        self.val_ds_class = ValUnbalancedMrArt
        self.train_ds_class = TrainUnbalancedMrArt
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
                with torch.autocast(device_type="cuda"):
                    if isinstance(batch["data"], torch.IntTensor):
                        batch["data"] = batch["data"].as_tensor()
                    batch["data"] = batch["data"].unsqueeze(0).cuda()

                    batch["data"] = (
                        self.pretrained_model(batch["data"]).cpu().squeeze(0)
                    )
                    cache_ds.append(batch)
        logging.error(f"Output shape {cache_ds[0]['data'].shape}")
        return cache_ds