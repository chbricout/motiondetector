"""Module to use Monte Carlo Dropout on our models"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from comet_ml import ExistingExperiment, APIExperiment
from tqdm import tqdm
from lightning import LightningModule
from src.network.archi import Model


def bincount2d(arr: np.ndarray, bins: int | None = None) -> np.ndarray:
    """Count number of occurence of each bin / class in a list
    of prediction for the same volume

    Args:
        arr (np.ndarray): two dimensional array (N_data_point, N_predictions)
         containing multiple list of predictions (one by data point)
        bins (int | None, optional): Number of bins / classes. Defaults to None.

    Returns:
        np.ndarray: Count of occurence for each bins (N_data_point, N_bins)
    """
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


def evaluate_mcdropout(
    model: Model,
    dataloader: DataLoader,
    experiment: ExistingExperiment | APIExperiment | None = None,
    label: str = "label",
    n_samples: int = 100,
) -> pd.DataFrame:
    """Evaluate Monte Carlo Dropout bin count for finetune models

    Args:
        model (Model): Trained model
        dataloader (DataLoader): Dataloader to use
        experiment (ExistingExperiment | APIExperiment | None , optional):
         Comet experiment to log on. Defaults to None.
        label (str, optional): label key in dataloader. Defaults to "label".
        n_samples (int, optional): number of prediction for MC Dropout.
          Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe with full results (logged on comet if experiment)
    """
    model.mc_dropout()
    res = []
    labels: list[int] = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):
            sample_pred = []
            labels = []
            for idx, batch in enumerate(dataloader):
                batch["data"] = batch["data"].cuda()
                labels += batch[label].tolist()
                sample_pred.append(torch.as_tensor(model.predict_step(batch, idx)))
                torch.cuda.empty_cache()
            res.append(torch.concat(sample_pred).unsqueeze(1))
    preds = torch.concat(res, 1).float()
    print(preds, labels)
    mean = torch.mean(preds, dim=1)
    std = torch.std(preds, dim=1)
    count = bincount2d(preds.int())

    print(mean, std, count)
    full_np = np.array([mean.tolist(), std.tolist(), labels])
    full = pd.DataFrame(full_np.T, columns=["mean", "std", "labels"])
    full["count"] = count.tolist()

    if experiment is not None:
        experiment.log_table("mcdropout-res.csv", full)

    return full


def pretrain_mcdropout(
    pl_module: LightningModule,
    dataloader: DataLoader,
    experiment: ExistingExperiment | APIExperiment | None = None,
    label: str = "label",
    n_samples: int = 100,
) -> pd.DataFrame:
    """Evaluate Monte Carlo Dropout for pretrain models

    Args:
        pl_module (LightningModule): Pretrained model
        dataloader (DataLoader): Dataloader to use
        experiment (ExistingExperiment | APIExperiment | None, optional):
            Comet experiment to log on. Defaults to None.
        label (str, optional): label key in dataloader. Defaults to "label".
        n_samples (int, optional): number of prediction for MC Dropout. Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe with full results (logged on comet if experiment)
    """
    pl_module.model.mc_dropout()
    res: list[torch.Tensor] = []
    labels: list[float | int] = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):
            sample_pred = []
            labels = []
            for idx, batch in enumerate(dataloader):
                batch["data"] = batch["data"].cuda()
                labels += batch[label].tolist()
                sample_pred.append(torch.as_tensor(pl_module.predict_step(batch, idx)))
                torch.cuda.empty_cache()
            sample_pred_tensor = torch.concat(sample_pred)
            res.append(sample_pred_tensor.unsqueeze(1))
    preds = torch.concat(res, 1).float()
    mean = torch.mean(preds, dim=1)
    std = torch.std(preds, dim=1)

    print(mean, std)
    full_np = np.array([mean.tolist(), std.tolist(), labels])
    full = pd.DataFrame(full_np.T, columns=["mean", "std", "labels"])

    if experiment is not None:
        experiment.log_table("mcdropout-res.csv", full)

    return full
