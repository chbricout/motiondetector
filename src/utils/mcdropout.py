"""Module to use Monte Carlo Dropout on our models"""

import logging

from comet_ml import ExistingExperiment, APIExperiment
import torch
from torch.utils.data import DataLoader
from lightning import LightningModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sb
from tqdm import tqdm
from src import config
from src.utils.comet import log_figure_comet
from src.utils.log import log_figure


def predict_mcdropout(
    pl_module: LightningModule,
    dataloader: DataLoader,
    n_preds: int = 100,
    label: str = "label",
) -> tuple[torch.Tensor, list[int | float], list[str]]:
    """Compute MC Dropout inference for any models

    Args:
        model (Model): Trained model
        dataloader (DataLoader): Dataloader to use
        n_samples (int, optional): number of prediction for MC Dropout.
          Defaults to 100.
        label (str, optional): label key in dataloader. Defaults to "label".

    Returns:
        tuple[torch.Tensor, list[int | float], list[str]]:
            - Prediction tensor (#dataset, n_samples)
            - Ground truth (#dataset)
            - Volumes identifiers (#dataset)
    """

    pl_module.model.mc_dropout()
    pl_module.cuda()
    res = []
    labels: list[int] = []
    with torch.no_grad():
        for _ in tqdm(range(n_preds)):
            sample_pred = []
            labels = []
            identifiers: list[str] = []
            for idx, batch in enumerate(dataloader):
                batch["data"] = batch["data"].cuda()
                labels += batch[label].tolist()
                identifiers += batch["identifier"]

                sample_pred.append(
                    torch.as_tensor(pl_module.predict_step(batch, idx)).cpu()
                )
                torch.cuda.empty_cache()
            res.append(torch.concat(sample_pred).unsqueeze(1))
    preds = torch.concat(res, 1).float()

    return preds, labels, identifiers


def bincount2d(arr: np.ndarray | torch.Tensor, bins: int | None = None) -> np.ndarray:
    """Count number of occurence of each bin / class in a list
    of prediction for the same volume

    Args:
        arr (np.ndarray | torch.Tensor): two dimensional array (N_data_point, N_predictions)
         containing multiple list of predictions (one by data point)
        bins (int | None, optional): Number of bins / classes. Defaults to None.

    Returns:
        np.ndarray: Count of occurence for each bins (N_data_point, N_bins)
    """
    if torch.is_tensor(arr):
        arr = arr.numpy()
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


def finetune_pred_to_df(
    preds: torch.Tensor, labels: list[int], identifiers: list[str]
) -> pd.DataFrame:
    """Transform Tensor of prediction in Dataframe for later analysis.
    Perform mean, std and count/bins.

    Args:
        preds (torch.Tensor): Prediction with shape (N_samples * N_predictions)
        labels (list[int]): True labels
        identifiers (list[str]): Volumes identifiers

    Returns:
        pd.DataFrame: Dataframe with: mean, std, label, count, predictions list,
        max_classe and confidence
    """
    mean = torch.mean(preds, dim=1, dtype=float)
    std = torch.std(preds, dim=1)
    count = bincount2d(preds.int())

    logging.debug(mean, std, count)
    np_concat = np.array([identifiers, mean.tolist(), std.tolist(), labels])
    df = pd.DataFrame(np_concat.T, columns=["identifier", "mean", "std", "label"])
    df["count"] = count.tolist()
    df["predictions"] = preds.tolist()
    df["max_classe"] = np.argmax(count)
    df["confidence"] = np.max(count) / np.sum(count)
    return df


def pretrain_pred_to_df(
    preds: torch.Tensor, labels: list[float | int], identifiers: list[str]
) -> pd.DataFrame:
    """Transform Tensor of prediction in Dataframe for later analysis.
    Perform mean and std.

    Args:
        preds (torch.Tensor): Prediction with shape (N_samples * N_predictions)
        labels (list[float | int]): True labels
        identifiers (list[str]): Volumes identifiers

    Returns:
        pd.DataFrame: Dataframe with: mean, std, label and predictions
    """
    mean = torch.mean(preds, dim=1, dtype=float)
    std = torch.std(preds, dim=1)

    logging.debug(mean, std)
    np_concat = np.array([identifiers, mean.tolist(), std.tolist(), labels])
    df = pd.DataFrame(np_concat.T, columns=["identifier", "mean", "std", "label"])
    df["predictions"] = preds.tolist()
    return df


def get_acc_prop(df: pd.DataFrame, confidence: float) -> tuple[float, float]:
    """Compute accuracy by keeping only prediction with confidence higher than `confidence`

    Args:
        df (pd.DataFrame): Dataframe with colums "label", "max_classe" and "confidence"
        confidence (int): lower acceptable confidence (included)

    Returns:
        tuple[float, float]: accuracy, proportion of original dataset kept
    """
    assert set(["confidence", "label", "max_classe"]).issubset(df.columns)

    filtered = df[df["confidence"] >= confidence]
    if len(filtered) > 0:
        filtered_accuracy = (
            filtered["label"] == filtered["max_classe"].astype(int)
        ).sum() / len(filtered)
    else:
        filtered_accuracy = 0
    filtered_prop = len(filtered) / len(df)
    return filtered_accuracy, filtered_prop


def finetune_confidence_plots(
    df: pd.DataFrame, filt_conf: float = config.CONFIDENCE_FILTER
) -> tuple[Figure, Figure]:
    """Plot finetuning confidence/accurac/proportion plot and
    swarmplot of prediction at confidence > `filt_conf`

    Args:
        df (pd.DataFrame): Dataframe from finetune MC-Dropout
        filt_conf (float): Confidence for swarmplot filter

    Returns:
        tuple[Figure, Figure]: Confidence plot, Swarm filter plot
    """
    accs = []
    props = []
    x = np.arange(0, 1, 0.01)
    for i in x:
        (a, p) = get_acc_prop(df, confidence=i)
        accs.append(a)
        props.append(p)

    confidence_fig = plt.figure(figsize=(6, 5))
    confidence = confidence_fig.add_subplot(1, 1, 1)
    confidence.plot(x, accs, label="accuracy")
    confidence.plot(x, props, label="kept proportion")
    confidence.set_xlabel("Confidence threshold (%)")
    confidence.legend()

    filtered_fig = plt.figure(figsize=(6, 5))
    filtered = filtered_fig.add_subplot(1, 1, 1)

    sb.swarmplot(
        df[df["confidence"] >= filt_conf], x="label", y="max_classe", ax=filtered
    )
    return confidence_fig, filtered_fig


def finetune_mcdropout(
    pl_module: LightningModule,
    dataloader: DataLoader,
    experiment: ExistingExperiment | APIExperiment | None = None,
    label: str = "label",
    n_preds: int = 100,
    log_figs: bool = True,
) -> pd.DataFrame:
    """Evaluate Monte Carlo Dropout bin count for finetune models

    Args:
        model (Model): Trained model
        dataloader (DataLoader): Dataloader to use
        experiment (ExistingExperiment | APIExperiment | None , optional):
         Comet experiment to log on. Defaults to None.
        label (str, optional): label key in dataloader. Defaults to "label".
        n_preds (int, optional): number of prediction for MC Dropout.
          Defaults to 100.
        log_figs (bool, optional): Flag to log figure. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe with full results (logged on comet if experiment)
    """
    mcdrop_res = predict_mcdropout(
        pl_module=pl_module, dataloader=dataloader, n_preds=n_preds, label=label
    )
    df = finetune_pred_to_df(*mcdrop_res)
    confidence_fig, filtered_fig = finetune_confidence_plots(df)

    if experiment is not None and log_figs:
        experiment.log_table("mcdropout-res.csv", df)
        log_figure_comet(confidence_fig, "confidence", experiment)
        log_figure_comet(filtered_fig, "filtered", experiment)
    elif log_figs:
        log_figure(
            confidence_fig,
            f"{pl_module.model.__class__.__name__}_{pl_module.__class__.__name__}",
            "confidence",
        )
        log_figure(
            filtered_fig,
            f"{pl_module.model.__class__.__name__}_{pl_module.__class__.__name__}",
            "filtered",
        )

    return df


def pretrain_mcdropout(
    pl_module: LightningModule,
    dataloader: DataLoader,
    experiment: ExistingExperiment | APIExperiment | None = None,
    label: str = "label",
    n_preds: int = 100,
) -> pd.DataFrame:
    """Evaluate Monte Carlo Dropout for pretrain models

    Args:
        pl_module (LightningModule): Pretrained model
        dataloader (DataLoader): Dataloader to use
        experiment (ExistingExperiment | APIExperiment | None, optional):
            Comet experiment to log on. Defaults to None.
        label (str, optional): label key in dataloader. Defaults to "label".
        n_preds (int, optional): number of prediction for MC Dropout. Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe with full results (logged on comet if experiment)
    """
    mcdrop_res = predict_mcdropout(
        pl_module=pl_module, dataloader=dataloader, n_preds=n_preds, label=label
    )
    df = pretrain_pred_to_df(*mcdrop_res)

    if experiment is not None:
        experiment.log_table("mcdropout-res.csv", df)

    return df
