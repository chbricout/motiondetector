import glob
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from comet_ml import API

from src import config

path_to_test = "test_report"
task_hue_order = ("SSIM", "MOTION", "BINARY")
api = API(config.COMET_API_KEY)


def ridge_plot(df, x, y):
    sb.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sb.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sb.FacetGrid(df, row=y, hue=y, aspect=10, height=1, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sb.kdeplot, x, bw_adjust=0.75, clip_on=False, fill=True, alpha=1, linewidth=1.5
    )
    g.map(sb.kdeplot, x, clip_on=False, color="w", lw=2, bw_adjust=0.75)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, x)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    sb.set_theme()


def retrieve_thresholds(unbalanced=False):
    mrart_results = []
    directory = (
        path.join(path_to_test, "unbalanced", "pretrain", "*")
        if unbalanced
        else path.join(path_to_test, "pretraining", "*")
    )
    for models_directory in glob.glob(directory):
        model, task, run = path.basename(models_directory).split("-")

        mrart = pd.read_csv(
            path.join(
                models_directory,
                "unbalanced-mrart_recap.csv" if unbalanced else "mrart_recap.csv",
            )
        )
        mrart["model"] = model
        mrart["task"] = task
        mrart["run_num"] = run

        mrart_results.append(
            mrart[
                [
                    "model",
                    "task",
                    "run_num",
                    "balanced_accuracy",
                    "threshold_1",
                    "threshold_2",
                    "f1_0",
                    "f1_1",
                    "f1_2",
                ]
            ]
        )

    return (
        pd.concat(mrart_results)
        .sort_values(["model", "task"], ascending=False)
        .reset_index(drop=True)
    )


def retrieve_pretrain():
    motion_res = []
    ssim_res = []
    binary_res = []

    for models_directory in glob.glob(path.join(path_to_test, "pretraining", "*")):
        model, task, *_ = path.basename(models_directory).split("-")

        results = pd.read_csv(path.join(models_directory, "results.csv"))
        results["model"] = model
        results["task"] = task
        if task == "MOTION":
            motion_res.append(results[["model", "task", "source", "r2", "rmse"]])
        elif task == "SSIM":
            ssim_res.append(results[["model", "task", "source", "r2", "rmse"]])
        elif task == "BINARY":
            binary_res.append(
                results[["model", "task", "source", "balanced_accuracy", "rmse"]]
            )

    motion_res = pd.concat(motion_res)
    ssim_res = pd.concat(ssim_res)
    binary_res = pd.concat(binary_res)
    motion_res = motion_res[motion_res["source"] == "simple"]
    ssim_res = ssim_res[ssim_res["source"] == "simple"]
    binary_res = binary_res[binary_res["source"] == "simple"]

    return motion_res, ssim_res, binary_res


def retrieve_transfer(unbalanced=False):
    full_results = []
    directory = (
        path.join(path_to_test, "ubalanced", "transfer", "*")
        if unbalanced
        else path.join(path_to_test, "transfer", "*")
    )
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


def retrieve_scratch(unbalanced=False):
    full_results = []
    directory = (
        path.join(path_to_test, "ubalanced", "scratch", "*")
        if unbalanced
        else path.join(path_to_test, "scratch", "*")
    )
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


def compute_norms(df):
    df["balanced_accuracy_minmax"] = (
        df["balanced_accuracy"] - df["balanced_accuracy"].min()
    ) / (df["balanced_accuracy"].max() - df["balanced_accuracy"].min())
    # df['balanced_accuracy_stand'] = (df['balanced_accuracy']-df['balanced_accuracy'].mean())/(df['balanced_accuracy'].std())


def min_max_norm(serie):
    return (serie - serie.min()) / (serie.max() - serie.min())


def ms_to_time(millis: int):
    seconds = millis // 1000
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return int(hours), int(minutes), int(seconds)


def get_duration_df_pretrain(experiments):
    durations = []
    for exp in experiments:
        ms = exp.get_metadata()["durationMillis"]
        model = exp.get_parameters_summary("model_class")["valueMax"]
        task = exp.get_parameters_summary("task")["valueMax"]
        hours, minutes = ms_to_time(ms)
        durations.append(
            (model, task, hours, minutes, f"{hours}:{str(minutes).zfill(2)}")
        )
    return pd.DataFrame(
        durations, columns=("model", "task", "hours", "minutes", "duration")
    )


def get_compute_usage_df(experiments, task_dependent=False):
    durations = []
    for exp in experiments:
        ms = exp.get_metadata()["durationMillis"]
        model = exp.get_parameters_summary("model")["valueMax"]

        hours, minutes = ms_to_time(ms)
        gpu_memory = (
            float(exp.get_metrics_summary("sys.gpu.0.used_memory")["valueMax"]) / 1e9
        )
        gpu_power_usage = (
            float(exp.get_metrics_summary("sys.gpu.0.power_usage")["valueMax"]) / 1000
        )

        if task_dependent:
            _, _, _, task, _ = exp.name.split("-")
            durations.append(
                (
                    model,
                    task,
                    ms,
                    hours,
                    minutes,
                    f"{hours}:{str(minutes).zfill(2)}",
                    gpu_memory,
                    gpu_power_usage,
                )
            )
        else:
            durations.append(
                (
                    model,
                    ms,
                    hours,
                    minutes,
                    f"{hours}:{str(minutes).zfill(2)}",
                    gpu_memory,
                    gpu_power_usage,
                )
            )

    if task_dependent:
        columns = (
            "model",
            "task",
            "millis",
            "hours",
            "minutes",
            "duration",
            "max_gpu_ram_used",
            "max_gpu_power_usage",
        )
    else:
        columns = (
            "model",
            "millis",
            "hours",
            "minutes",
            "duration",
            "max_gpu_ram_used",
            "max_gpu_power_usage",
        )

    return pd.DataFrame(durations, columns=columns)
