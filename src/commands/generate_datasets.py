"""
Module to generate synthetic motion datasets used for pretraining
"""

import logging
import os
import shutil
import tarfile
from typing import Type

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import pandas as pd

from src.dataset.ampscz.ampscz_dataset import (
    PretrainTrainAMPSCZ,
    PretrainValAMPSCZ,
    PretrainTestAMPSCZ,
)
from src.dataset.base_dataset import BaseDataset
from src.dataset.hcpep.hcpep_dataset import TrainHCPEP, ValHCPEP, TestHCPEP
from src.transforms.generate import (
    Preprocess,
    CreateSynthVolume,
    FinalCrop,
    SyntheticPipeline,
)
from src import config


def setup_dataset_tree(dataset_dir: str, modes: list[str]):
    """Create every folder needed for the new dataset (if needed) and remove previous
    generation attempt if the main folder already exists

    Args:
        dataset_dir (str): Path to the new dataset
        modes (list[str]): list of mode to create folder for.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    for mode in modes:
        if os.path.exists(f"{dataset_dir}/{mode}"):
            shutil.rmtree(f"{dataset_dir}/{mode}")
        os.mkdir(f"{dataset_dir}/{mode}")


def load_data(datasets: list[Type[BaseDataset]]) -> Dataset:
    """load data from every dataset in advance and return
    a concatenated dataset ready for synthethic generation

    Args:
        datasets (list[Type[BaseDataset]]): list of datasets to use

    Returns:
        Dataset: Concatenated loaded dataset
    """
    load_tsf = Preprocess()
    loaded = [ds.from_env(load_tsf) for ds in datasets]
    return ConcatDataset(loaded)


def generate_data(dataset: Dataset, dataset_dir: str, mode: str, num_iter: int):
    """Generate a synthetic dataset by processing and storing
    the whole dataloader `n_iter` times.
    Store file path and artifacts labels to `csv_path`.

    Args:
        dataloader (DataLoader): Dataloader with synthetic transform
        dataset_dir (str): Directory to store dataset.
        mode (str): Dataset's mode (train, val, test)
        num_iter (int): Number of iteration for generation
    """
    lst_dict = []
    synth_pipeline = SyntheticPipeline(dataset_dir=dataset_dir, mode=mode)
    synth_dataset = Dataset(dataset, transform=synth_pipeline)
    dataloader = DataLoader(
        synth_dataset,
        batch_size=1,
        num_workers=64,
        prefetch_factor=2,
    )
    for i in tqdm(range(num_iter)):
        synth_pipeline.iteration = i
        for element in tqdm(dataloader):
            lst_dict.append({
            "data": element['data'][0],
            "motion_mm": element["motion_mm"].item(),
            "ssim_loss": element["ssim_loss"].item(),
            "motion_binary": element["motion_binary"].item(),
            "identifier": element['identifier'][0],
            "group": element['group'][0],
        })
    
    pd.DataFrame.from_records(lst_dict).to_csv(f"{dataset_dir}/{mode}.csv")
    synth_pipeline.save_parameters()


def generated_to_tar(root_dir: str, new_dataset: str, to_archive: str):
    """Archive generated files for future usage

    Args:
        root_dir (str): Root dir of all datasets
        new_dataset (str): New dataset name
        to_archive (str): path to generated dataset
    """
    filename = "generate_dataset.tar"
    dataset_dir = os.path.join(root_dir, new_dataset + "_archive")
    os.makedirs(dataset_dir, exist_ok=True)

    logging.info("Writing dataset to tar archive")
    with tarfile.open(os.path.join(dataset_dir, filename), "w") as tar_obj:
        tar_obj.add(to_archive)
    logging.info("Dataset archived, remember to remove folder")


def launch_generate_data(new_dataset: str):
    """Generate synthetic motion dataset and store everything (Volumes and CSVs)

    Args:
        new_dataset (str): Name of the new dataset
    """
    root_dir = config.GENERATE_ROOT_DIR
    logging.info("root dataset is %s", root_dir)
    dataset_dir = os.path.join(root_dir, new_dataset)

    confs = {
        "test": [PretrainTestAMPSCZ, TestHCPEP],
        "val": [PretrainValAMPSCZ, ValHCPEP],
        "train": [PretrainTrainAMPSCZ, TrainHCPEP],
    }

    setup_dataset_tree(dataset_dir, modes=confs.keys())

    for mode, datasets in confs.items():
        logging.info("Creating dataset for mode : %s", mode)
        loaded_ds = load_data(datasets)
        generate_data(
            dataset=loaded_ds,
            dataset_dir=dataset_dir,
            mode=mode,
            num_iter=300,
        )
    generated_to_tar(root_dir, new_dataset, dataset_dir)
