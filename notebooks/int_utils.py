import glob
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from comet_ml import API

from src import config

path_to_test = "report"
api = API(config.COMET_API_KEY)


def retrieve_transfer():
    full_results = []
    directory = path.join(path_to_test, "transfer", "*")
    for models_directory in glob.glob(directory):
        model, task, run_num = path.basename(models_directory).split("-")

        results = pd.read_csv(path.join(models_directory, "results.csv"))
        results["model"] = model
        results["task"] = task
        results["run_num"] = int(run_num)

        full_results.append(
            results[
                [
                    "model",
                    "task",
                    "run_num",
                    "balanced_accuracy",
                    "f1_0",
                    "f1_1",
                    "f1_2",
                ]
            ]
        )

    full_results = pd.concat(full_results)
    return full_results.sort_values(
        ["model", "task", "run_num"], ascending=False
    ).reset_index(drop=True)


def retrieve_scratch():
    full_results = []
    directory = path.join(path_to_test, "scratch", "*")
    for models_directory in glob.glob(directory):
        model, run_num = (
            path.basename(models_directory).removesuffix(".ckpt").split("-")
        )

        results = pd.read_csv(path.join(models_directory, "results.csv"))
        results["model"] = model
        results["run_num"] = int(run_num)

        full_results.append(
            results[
                [
                    "model",
                    "run_num",
                    "balanced_accuracy",
                    "f1_0",
                    "f1_1",
                    "f1_2",
                ]
            ]
        )

    full_results = pd.concat(full_results)
    full_results.sort_values("balanced_accuracy", ascending=False)
    return full_results


def ms_to_time(millis: int):
    seconds = millis // 1000
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return int(hours), int(minutes), int(seconds)


def get_compute_usage_df(experiments, task_dependent=False, pretrain=False):
    durations = []
    for exp in experiments:
        ms = exp.get_metadata()["durationMillis"]
        model = exp.get_parameters_summary("model")["valueMax"]

        hours, minutes, seconds = ms_to_time(ms)
        gpu_memory = (
            float(exp.get_metrics_summary("sys.gpu.0.used_memory")["valueMax"]) / 1e9
        )
        gpu_power_usage = (
            float(exp.get_metrics_summary("sys.gpu.0.power_usage")["valueMax"]) / 1000
        )

        if task_dependent:
            if pretrain:
                _, task, _, _ = exp.name.split("-")
            else:
                _, _, task, _ = exp.name.split("-")
            durations.append(
                (
                    model,
                    task,
                    ms,
                    hours,
                    minutes,
                    seconds,
                    f"{hours}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}",
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
                    seconds,
                    f"{hours}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}",
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
            "seconds",
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
            "seconds",
            "duration",
            "max_gpu_ram_used",
            "max_gpu_power_usage",
        )

    return pd.DataFrame(durations, columns=columns)
