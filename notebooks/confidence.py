# %%
import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from src import config

path_to_test = "test_report"


def retrieve_transfer():
    full_results = []
    directory = path.join(path_to_test, "transfer", "*")

    for models_directory in glob.glob(directory):
        model, task, run_num = path.basename(models_directory).split("-")

        results = pd.read_csv(path.join(models_directory, "mrart_recap.csv"))
        results["model"] = model
        results["task"] = task
        results["run_num"] = int(run_num)

        full_results.append(
            results[
                [
                    "model",
                    "task",
                    "run_num",
                    "source",
                    "balanced_accuracy",
                    "f1_0",
                    "f1_1",
                    "f1_2",
                ]
            ]
        )

    full_results = pd.concat(full_results)
    simple_res = full_results[full_results["source"] == "simple"]
    simple_res = simple_res.drop(columns="source")
    return simple_res.sort_values(
        ["model", "task", "run_num"], ascending=False
    ).reset_index(drop=True)


def retrieve_scratch():
    full_results = []
    directory = path.join(path_to_test, "scratch", "*")

    for models_directory in glob.glob(directory):
        model, run_num = (
            path.basename(models_directory).removesuffix(".ckpt").split("-")
        )

        results = pd.read_csv(path.join(models_directory, "mrart_recap.csv"))
        results["model"] = model
        results["run_num"] = int(run_num)

        full_results.append(
            results[
                [
                    "model",
                    "run_num",
                    "source",
                    "balanced_accuracy",
                    "f1_0",
                    "f1_1",
                    "f1_2",
                ]
            ]
        )

    full_results = pd.concat(full_results)
    simple_scratch = full_results[full_results["source"] == "simple"]
    simple_scratch.sort_values("balanced_accuracy", ascending=False)
    return simple_scratch


def retrieve_confidence(setting: str):
    conf_results = []
    directory = path.join(path_to_test, setting, "*")

    for models_directory in glob.glob(directory):
        if setting == "transfer":
            model, task, run_num = path.basename(models_directory).split("-")
        else:
            model, run_num = (
                path.basename(models_directory).removesuffix(".ckpt").split("-")
            )
            task = ""

        results = pd.read_csv(path.join(models_directory, "confidence.csv"))

        for thresh in np.linspace(0.5, 0.99, 50):
            selected_row = results[
                results["threshold_confidence"] > (thresh - 1e-4)
            ].iloc[0]
            conf_results.append(
                [
                    model,
                    int(run_num),
                    task,
                    selected_row["threshold_confidence"],
                    selected_row["balanced_accuracy"],
                    selected_row["kept_proportion"],
                ]
            )

    conf_results = pd.DataFrame(
        conf_results,
        columns=[
            "model",
            "run_num",
            "task",
            "source",
            "balanced_accuracy",
            "kept_proportion",
        ],
    )
    return conf_results


# %%
scratch_df = retrieve_confidence("scratch")
transfer_df = retrieve_confidence("transfer")

# %%
scratch_df.iloc[scratch_df["balanced_accuracy"].argmax()]
transfer_df.iloc[transfer_df["balanced_accuracy"].argmax()]

# %%
scratch_df["id"] = scratch_df["model"] + "-" + scratch_df["run_num"].astype(str)
sb.lineplot(
    scratch_df,
    y="balanced_accuracy",
    x="source",
    hue="model",
    estimator=None,
    units="run_num",
)
plt.ylim((0.3, 1.05))

# %%
task_df = transfer_df[transfer_df["task"] == "SSIM"]
task_df["id"] = task_df["model"] + "-" + task_df["task"].astype(str)
sb.lineplot(
    task_df,
    y="balanced_accuracy",
    x="source",
    hue="id",
    estimator=None,
    units="run_num",
)
plt.ylim((0.3, 1.05))

# %%
test = pd.read_csv("test_report/scratch/CONV5_FC3-2/dropout.csv")
test[test["confidence"] > 0.94999]


# %%


def compute_lower(df):
    df
