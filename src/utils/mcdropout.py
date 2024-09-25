"""Module to use Monte Carlo Dropout on our models"""

import logging

from comet_ml import ExistingExperiment, APIExperiment
import torch
from torch.utils.data import DataLoader
from monai.data import MetaTensor
from lightning import LightningModule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sb
from tqdm import tqdm
from src import config
from src.utils.comet import log_figure_comet
from src.utils.confidence import confidence_finetune
from src.utils.log import log_figure
from src.utils.task import label_from_task_class
import src.utils.confidence as conf


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
    encoder = torch.compile(pl_module.encode)
    classifier = torch.compile(pl_module.classify)
    res = []
    labels: list[int] = []
    with torch.no_grad():
        cache_ds = []
        labels = []
        identifiers: list[str] = []
        for idx, batch in enumerate(dataloader):
            labels += batch[label].tolist()
            identifiers += batch["identifier"]

            with torch.autocast(device_type="cuda"):
                if isinstance(batch["data"], MetaTensor):
                    batch["data"] = batch["data"].as_tensor()
                batch["data"] = batch["data"].cuda()

                encoding = encoder(batch["data"])
                cache_ds.append(encoding.cpu())

        for _ in tqdm(range(n_preds)):
            sample_pred = []
            for encoding in cache_ds:
                with torch.autocast(device_type="cuda"):
                    batch_pred = classifier(encoding.cuda())
                    sample_pred.append(batch_pred.cpu())
            res.append(torch.concat(sample_pred).unsqueeze(1))
    preds = torch.concat(res, 1).float()

    return preds, labels, identifiers


def bincount2d(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    """Count number of occurence of each bin / class in a list
    of prediction for the same volume

    Args:
        arr (np.ndarray | torch.Tensor): two dimensional array (N_data_point, N_predictions)
         containing multiple list of predictions (one by data point)

    Returns:
        np.ndarray: Count of occurence for each bins (N_data_point, N_bins)
    """
    if torch.is_tensor(arr):
        arr = arr.numpy()
    count= np.apply_along_axis(np.bincount, axis=1, arr= arr,
                                          minlength = np.max(arr) +1)

    return count


def finetune_pred_to_df(
    preds: torch.Tensor, labels: list[int | float], identifiers: list[str]
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
    df["max_classe"] = np.argmax(count, axis=1)
    df["confidence"] = np.max(count, axis=1) / np.sum(count, axis=1)
    df["label"] = df["label"].astype(int)

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
    logging.debug("mean %f \nstd %f std", mean, std)
    np_concat = np.array([identifiers, mean.tolist(), std.tolist(), labels])
    df = pd.DataFrame(np_concat.T, columns=["identifier", "mean", "std", "label"])
    df["predictions"] = preds.tolist()
    df["mean"] = df["mean"].astype(float)
    df["std"] = df["std"].astype(float)
    df["label"] = df["label"].astype(float)

    return df


def finetune_filter_plot(
    df: pd.DataFrame, filt_conf: float = config.CONFIDENCE_FILTER
) -> tuple[Figure, Figure]:
    """Plot finetuning confidence/accurac/proportion plot and
    swarmplot of prediction at confidence > `filt_conf`

    Args:
        df (pd.DataFrame): Dataframe from finetune confidence
        filt_conf (float): Confidence for swarmplot filter

    Returns:
        Figure: Swarm filter plot
    """
    filtered_fig = plt.figure(figsize=(6, 5))
    filtered = filtered_fig.add_subplot(1, 1, 1)
    sb.stripplot(
        df[df["confidence"] >= filt_conf],
        x="label",
        y="max_classe",
        ax=filtered,
    )
    return filtered_fig


def transfer_mcdropout(
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
    conf_df = confidence_finetune(df)
    confidence_fig = conf.plot_confidence(
        conf_df=conf_df,
        threshold_label="threshold_confidence",
        metric_label="balanced_accuracy",
        threshold_axis="Threshold Confidence",
        metric_axis="Balanced Accuracy",
    )
    filtered_fig = finetune_filter_plot(df)

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

    return df, conf_df, confidence_fig, filtered_fig


def pretrain_mcdropout(
    pl_module: LightningModule,
    dataloader: DataLoader,
    experiment: ExistingExperiment | APIExperiment | None = None,
    label: str | None = None,
    n_preds: int = 100,
) -> pd.DataFrame:
    """Evaluate Monte Carlo Dropout for pretrain models

    Args:
        pl_module (LightningModule): Pretrained model
        dataloader (DataLoader): Dataloader to use
        experiment (ExistingExperiment | APIExperiment | None, optional):
            Comet experiment to log on. Defaults to None.
        label (str | None, optional): label key in dataloader. Defaults to None.
        n_preds (int, optional): number of prediction for MC Dropout. Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe with full results (logged on comet if experiment)
    """
    if label is None:
        label = label_from_task_class(pl_module.__class__)
    mcdrop_res = predict_mcdropout(
        pl_module=pl_module, dataloader=dataloader, n_preds=n_preds, label=label
    )
    df = pretrain_pred_to_df(*mcdrop_res)

    if experiment is not None:
        experiment.log_table("mcdropout-res.csv", df)

    return df
