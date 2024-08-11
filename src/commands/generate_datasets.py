"""
Module to generate synthetic motion datasets used for pretraining
"""

import os
import shutil

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms import SaveImage, Transform
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import pandas as pd

from src.dataset.ampscz.ampscz_dataset import PretrainTrainAMPSCZ, PretrainValAMPSCZ
from src.dataset.hcpep.hcpep_dataset import TrainHCPEP, ValHCPEP
from src.transforms.generate import Preprocess, CreateSynthVolume, FinalCrop


class SaveElement(Transform):
    """Special tranform to save the newly created volume in your intended folder"""

    def __init__(self, new_dataset: str, mode="train", iteration=0):
        super().__init__()
        self.mode = mode
        self.save = SaveImage(savepath_in_metadict=True, resample=False)
        self.iteration = iteration
        self.new_dataset = new_dataset
        self.base_path = f"/home/cbricout/scratch/{self.new_dataset}/{self.mode}/"

    def __call__(self, element):
        path = (
            self.base_path + f"{element['sub_id']}-{element['ses_id']}-{self.iteration}"
        )
        self.save(element["data"], filename=path)

        return {
            "data": element["data"],
            "motion_mm": element["motion_mm"],
            "ssim_loss": element["ssim_loss"],
            "motion_binary": element["motion_binary"],
            "sub_id": str(element["sub_id"]),
            "ses_id": str(element["ses_id"]),
        }


def setup_dataset_tree(new_dataset: str):
    """Create every folder needed for the new dataset (if needed) and remove previous
    generation attempt if the main folder already exists

    Args:
        new_dataset (str): Name of the new dataset
    """
    if not os.path.exists(f"/home/cbricout/scratch/{new_dataset}"):
        os.mkdir(f"/home/cbricout/scratch/{new_dataset}")
    else:
        if os.path.exists(f"/home/cbricout/scratch/{new_dataset}/train"):
            shutil.rmtree(f"/home/cbricout/scratch/{new_dataset}/train")
        if os.path.exists(f"/home/cbricout/scratch/{new_dataset}/val"):
            shutil.rmtree(f"/home/cbricout/scratch/{new_dataset}/val")
    os.mkdir(f"/home/cbricout/scratch/{new_dataset}/train")
    os.mkdir(f"/home/cbricout/scratch/{new_dataset}/val")


def launch_generate_data(new_dataset: str):
    """Generate synthetic motion dataset and store everything (Volumes and CSVs)

    Args:
        new_dataset (str): Name of the new dataset
    """
    setup_dataset_tree(new_dataset)

    load_tsf = Preprocess()
    synth_tsf = CreateSynthVolume(elastic_activate=True)
    crop_tsf = FinalCrop()
    save_train = SaveElement(new_dataset,"train")
    save_val = SaveElement(new_dataset,"val")

    val_ampscz_ds = PretrainValAMPSCZ.narval(load_tsf)
    val_hcpep_ds = ValHCPEP.narval(load_tsf)
    synth_val_ds = Dataset(
        data=ConcatDataset([val_ampscz_ds, val_hcpep_ds]),
        transform=Compose([synth_tsf, crop_tsf, save_val]),
    )

    lst_dict = []
    for i in tqdm(range(20)):
        save_val.iteration = i
        dataloader = DataLoader(synth_val_ds, batch_size=20, num_workers=30, prefetch_factor=15,
)
        for element in tqdm(dataloader):
            records = [dict(zip(element, t)) for t in zip(*element.values())]
            for r in records:
                new_dict = {
                    "motion_mm": r["motion_mm"].item(),
                    "ssim_loss": r["ssim_loss"].item(),
                    "motion_binary": r["motion_binary"],
                    "sub_id": r["sub_id"],
                    "ses_id": r["ses_id"],
                }
                new_dict["group"] = "val"
                new_dict["data"] = r["data"].meta["saved_to"]
                lst_dict.append(new_dict)
    pd.DataFrame.from_records(lst_dict).to_csv(
        f"/home/cbricout/scratch/{new_dataset}/val.csv"
    )

    train_ampscz_ds = PretrainTrainAMPSCZ.narval(load_tsf)
    train_hcpep_ds = TrainHCPEP.narval(load_tsf)
    synth_train_ds = Dataset(
        data=ConcatDataset([train_hcpep_ds, train_ampscz_ds]),
        transform=Compose([synth_tsf, crop_tsf, save_train]),
    )
    lst_dict = []
    for i in tqdm(range(20)):
        save_train.iteration = i
        dataloader = DataLoader(synth_train_ds, batch_size=20, num_workers=30, prefetch_factor=15,)
        for element in tqdm(dataloader):
            records = [dict(zip(element, t)) for t in zip(*element.values())]
            for r in records:
                new_dict = {
                    "motion_mm": r["motion_mm"].item(),
                    "ssim_loss": r["ssim_loss"].item(),
                    "motion_binary": r["motion_binary"],
                    "sub_id": r["sub_id"],
                    "ses_id": r["ses_id"],
                }
                new_dict["group"] = "train"
                new_dict["data"] = r["data"].meta["saved_to"]
                lst_dict.append(new_dict)

    pd.DataFrame.from_records(lst_dict).to_csv(
        f"/home/cbricout/scratch/{new_dataset}/train.csv"
    )
