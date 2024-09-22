"""
Module defining function related to confidence estimation
"""

from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_prop_metrics(
    df: pd.DataFrame,
    threshold: float,
    threshold_label: str,
    metric_f: Callable[[pd.DataFrame], float],
):
    filtered = df[df[threshold_label] <= threshold]
    if len(filtered) == 0:
        return 0, 0
    mse = metric_f(filtered)
    filtered_prop = len(filtered) / len(df)
    return filtered_prop, mse


def confidence_pretrain(df: pd.DataFrame) -> pd.DataFrame:
    res=[]
    for thresh in np.arange(0.00, 0.023, 0.001):
        prop, rmse = compute_prop_metrics(
            df,
            thresh,
            threshold_label="std",
            metric_f=lambda x: mean_squared_error(
                df["mean"], df["label"], squared=False
            ),
        )
        res.append((thresh, prop, rmse))
    
    return_df = pd.DataFrame([], columns=["threshold_std", "kept_proportion", "rmse"])
    print(return_df)
    return return_df
