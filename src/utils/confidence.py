"""
Module defining function related to confidence estimation
"""

from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error


def compute_prop_metrics(
    df: pd.DataFrame,
    threshold: float,
    threshold_label: str,
    metric_f: Callable[[pd.DataFrame], float],
    threshold_f : Callable[[pd.Series, float], pd.Series],
):
    filtered = df[threshold_f(df[threshold_label], threshold)]
    if len(filtered) == 0:
        return 0, 0
    mse = metric_f(filtered)
    filtered_prop = len(filtered) / len(df)
    return filtered_prop, mse


def plot_confidence(
    conf_df: pd.DataFrame,
    threshold_label: str,
    metric_label: str,
    threshold_axis: str = "Threshold",
    metric_axis: str = "Metric",
):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(
        conf_df[threshold_label], conf_df[metric_label], color="red", label=metric_axis
    )
    ax2.plot(
        conf_df[threshold_label],
        conf_df["kept_proportion"],
        color="blue",
        label="Kept proportion",
    )
    ax.set_xlabel(threshold_axis)
    ax.set_ylabel(metric_axis)
    ax2.set_ylabel("Kept Proportion (%)")
    ax2.legend(
        handles=[a.lines[0] for a in [ax, ax2]],
        labels=[metric_axis, "Kept Proportion (%)"],
    )
    return fig


def confidence_pretrain(df: pd.DataFrame) -> pd.DataFrame:
    res = []
    min_std = df["std"].min()
    max_std = df["std"].max()
    for thresh in np.arange(min_std, max_std, (max_std - min_std) / 100):
        prop, rmse = compute_prop_metrics(
            df,
            thresh,
            threshold_label="std",
            metric_f=lambda x: mean_squared_error(x["label"], x["mean"], squared=False),
            threshold_f=lambda serie, x : serie <= x
        )
        res.append((thresh, prop, rmse))

    return_df = pd.DataFrame(res, columns=["threshold_std", "kept_proportion", "rmse"])
    return return_df


def confidence_finetune(df: pd.DataFrame) -> pd.DataFrame:
    res = []
    for thresh in np.arange(0, 1, 0.01):
        prop, baccuracy = compute_prop_metrics(
            df,
            thresh,
            threshold_label="confidence",
            metric_f=lambda x: balanced_accuracy_score(x["label"], x["max_classe"]),
            threshold_f=lambda serie, x : serie >= x
        )
        res.append((thresh, prop, baccuracy))

    return_df = pd.DataFrame(
        res, columns=["threshold_confidence", "kept_proportion", "balanced_accuracy"]
    )
    return return_df
