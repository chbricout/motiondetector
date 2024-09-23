import glob
from os import path
import os
import shutil

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
import seaborn as sb

from src.dataset.mrart.mrart_dataset import TestMrArt
from src.dataset.pretraining.pretraining_dataset import (
    PretrainTest,
    PretrainVal,
    PretrainTrain,
)
from src.training.pretrain_logic import PretrainingTask
from src.transforms.load import LoadSynth
from src.utils import task as task_utils
import src.training.eval as teval
from src.utils import mcdropout
from src.utils import confidence as conf


def test_pretrain_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        test_pretrain_model(ckpt_path=ckpt)


def load_from_ckpt(ckpt_path: str):
    base_name = path.basename(ckpt_path)
    _, task, *_ = base_name.split("-")
    task_class = task_utils.str_to_task(task)
    return task_class.load_from_checkpoint(checkpoint_path=ckpt_path), task


def test_pretrain_model(ckpt_path: str):
    module, task = load_from_ckpt(ckpt_path=ckpt_path)
    print("Start Evaluation")

    exp = path.basename(ckpt_path).split(".")[0]
    report_dir = path.join("test_report", exp)
    if path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    base_metrics = []
    for dataset, mode in [
        (PretrainTest, "test"),
        (PretrainVal, "val"),
    ]:
        print(f"Eval dataset : {mode}")
        ds = dataset.from_env(LoadSynth.from_task(task))
        ds.define_label(task)
        pretrain_dl = DataLoader(
            ds,
            batch_size=60,
            pin_memory=True,
            num_workers=20,
            prefetch_factor=2,
        )
        print("Predict pretrain")
        simple_pretrain_df = teval.get_pred_from_pretrain(
            module, pretrain_dl, mode, label=task_utils.label_from_task(task)
        )
        print("Pretrain DONE")

        print("Predict MC-Dropout")
        dropout_pretrain_df = mcdropout.pretrain_mcdropout(
            module, pretrain_dl, n_preds=1_000, label=task_utils.label_from_task(task)
        )
        print("MC-Dropout DONE")

        for df, source, pred_label in [
            (simple_pretrain_df, "simple", "pred"),
            (dropout_pretrain_df, "mcdropout", "mean"),
        ]:
            base_metrics.append(
                [
                    mode,
                    source,
                    r2_score(df["label"], df[pred_label]),
                    mean_squared_error(df["label"], df[pred_label], squared=False),
                ]
            )

        conf_df = conf.confidence_pretrain(dropout_pretrain_df)

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.plot(conf_df["threshold_std"], conf_df["rmse"], color="red", label="rmse")
        ax2.plot(
            conf_df["threshold_std"],
            conf_df["kept_proportion"],
            color="blue",
            label="Kept proportion",
        )

        plt.xlabel("Confidence threshold (Standard Deviation)")
        ax2.legend(
            handles=[a.lines[0] for a in [ax, ax2]],
            labels=["Root Mean Squared Error", "Kept Proportion (%)"],
        )
        plt.tight_layout()
        plt.savefig(path.join(report_dir, mode))

    base_metrics_df = pd.DataFrame(
        base_metrics, columns=["mode", "source", "r2", "rmse"]
    )
    base_metrics_df.to_csv(path.join(report_dir, "results.csv"))
