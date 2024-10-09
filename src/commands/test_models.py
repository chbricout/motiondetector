import glob
import os
import shutil
from os import path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

import src.training.eval as teval
from src import config
from src.dataset.base_dataset import BaseDataset
from src.dataset.mrart.mrart_dataset import (
    TestMrArt,
    TestUnbalancedMrArt,
    TrainMrArt,
    TrainUnbalancedMrArt,
)
from src.dataset.pretraining.pretraining_dataset import PretrainTest
from src.training.common_logic import BaseFinalTrain
from src.training.pretrain_logic import BinaryPretrainingTask, PretrainingTask
from src.training.scratch_logic import MRArtScratchTask
from src.training.transfer_logic import MrArtTransferTask
from src.transforms.load import FinetuneTransform, LoadSynth
from src.utils import confidence as conf
from src.utils import mcdropout, metrics
from src.utils import task as task_utils


def test_pretrain_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        module, task, report_dir = setup_test_pretrain(ckpt)
        test_pretrain_model_pretrain_data(
            module=module, task=task, report_dir=report_dir
        )
        test_pretrain_model_mrart_data(
            module=module,
            report_dir=report_dir,
            datasets=[("train", TrainMrArt), ("test", TestMrArt)],
            prefix="mrart",
        )
        test_pretrain_model_mrart_data(
            module=module,
            report_dir=report_dir,
            datasets=[("train", TrainUnbalancedMrArt), ("test", TestUnbalancedMrArt)],
            prefix="unbalanced-mrart",
        )


def setup_test_pretrain(ckpt_path: str) -> tuple[PretrainingTask, str, str]:
    module, task = task_utils.load_pretrain_from_ckpt(ckpt_path=ckpt_path)
    print(f"Start Evaluation for {ckpt_path}")

    exp = path.basename(ckpt_path).split(".")[0]
    report_dir = path.join("test_report", "pretraining", exp)
    if path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    return module, task, report_dir


def inference_test(
    module: PretrainingTask, dataset: Dataset, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataloader = DataLoader(
        dataset,
        batch_size=24,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=2,
    )
    print("Predict pretrain")
    simple_pretrain_df = teval.get_pred_from_pretrain(module, dataloader, label=label)
    print("Pretrain DONE")

    print("Predict MC-Dropout")
    drop_res = mcdropout.predict_mcdropout(module, dataloader, n_preds=10, label=label)
    convert_func = (
        mcdropout.finetune_pred_to_df
        if isinstance(module, BinaryPretrainingTask)
        else mcdropout.pretrain_pred_to_df
    )
    dropout_pretrain_df = convert_func(*drop_res)
    print("MC-Dropout DONE")

    return simple_pretrain_df, dropout_pretrain_df


def test_pretrain_model_pretrain_data(
    module: PretrainingTask, task: str, report_dir: str
):
    base_metrics = []
    is_binary = task == "BINARY"

    ds = PretrainTest.from_env(LoadSynth.from_task(task))
    ds.define_label(task)

    simple_df, dropout_df = inference_test(
        module, ds, "label" if is_binary else task_utils.label_from_task(task)
    )

    for df, source, pred_label in [
        (simple_df, "simple", "pred"),
        (dropout_df, "mcdropout", "mean"),
    ]:
        if task == "BINARY":
            base_metrics.append(
                [
                    source,
                    balanced_accuracy_score(df["label"], df[pred_label].round()),
                    mean_squared_error(
                        df["label"], df[pred_label].round(), squared=False
                    ),
                ]
            )
        else:
            base_metrics.append(
                [
                    source,
                    r2_score(df["label"], df[pred_label]),
                    mean_squared_error(df["label"], df[pred_label], squared=False),
                ]
            )
    confidence_func = (
        conf.confidence_finetune if is_binary else conf.confidence_pretrain
    )
    conf_df = confidence_func(dropout_df)

    fig = conf.plot_confidence(
        conf_df=conf_df,
        threshold_label="threshold_confidence" if is_binary else "threshold_std",
        metric_label="balanced_accuracy" if is_binary else "rmse",
        threshold_axis=(
            "Threshold Confidence" if is_binary else "Threshold Standard Deviation"
        ),
        metric_axis="Balanced Accuracy" if is_binary else "Root Mean Squared Error",
    )
    fig.savefig(path.join(report_dir, "test"))

    base_metrics_df = pd.DataFrame(
        base_metrics,
        columns=["source", "balanced_accuracy" if is_binary else "r2", "rmse"],
    )
    base_metrics_df.to_csv(path.join(report_dir, "results.csv"))
    conf_df.to_csv(path.join(report_dir, "confidence.csv"))
    plt.close()


def test_pretrain_model_mrart_data(
    module: PretrainingTask,
    report_dir: str,
    datasets: list[tuple[str, BaseDataset]],
    prefix: str,
):
    all_mode_df = []
    for mode, dataset in datasets:
        dl = DataLoader(dataset.from_env(FinetuneTransform()))
        all_mode_df.append(teval.get_pred_from_pretrain(module, dl, mode))
    all_mode_df = pd.concat(all_mode_df)
    acc, per_class_f1, thresholds, fig_thresh, cm_fig = teval.separation_capacity(
        all_mode_df, train_on="train", test_on="test"
    )

    fig_thresh.tight_layout()
    fig_thresh.savefig(path.join(report_dir, f"{prefix}-thresholds"))

    cm_fig.tight_layout()
    cm_fig.savefig(path.join(report_dir, f"{prefix}-confusion"))

    all_mode_df.to_csv(path.join(report_dir, f"{prefix}.csv"))

    recap = pd.DataFrame(
        [
            [
                acc,
                thresholds[0],
                thresholds[1],
                per_class_f1[0],
                per_class_f1[1],
                per_class_f1[2],
            ]
        ],
        columns=[
            "balanced_accuracy",
            "threshold_1",
            "threshold_2",
            "f1_0",
            "f1_1",
            "f1_2",
        ],
    )
    recap.to_csv(path.join(report_dir, f"{prefix}_recap.csv"))
    plt.close()


def test_scratch_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        print(f"Start Evaluation for {ckpt}")
        exp = path.basename(ckpt).split(".")[0]
        report_dir = path.join("test_report", "scratch", exp)
        if path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        module = MRArtScratchTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(module=module, report_dir=report_dir)


def test_scratch_model(
    module: BaseFinalTrain, report_dir: str, dataset_class: BaseDataset = TestMrArt
):

    dl = DataLoader(dataset_class.from_env(FinetuneTransform()))
    simple_df = teval.get_pred_from_pretrain(module, dl)
    dropout_df, conf_df, confidence_fig, filtered_fig = mcdropout.transfer_mcdropout(
        module, dl, n_preds=1000, log_figs=False
    )

    base_metrics = []
    for df, source, pred_label in [
        (simple_df, "simple", "pred"),
        (dropout_df, "mcdropout", "mean"),
    ]:
        acc, per_class_f1, cm_fig = metrics.prediction_report(
            df["label"], df[pred_label].astype(int)
        )
        base_metrics.append([source, acc, *per_class_f1])
        cm_fig.tight_layout()
        cm_fig.savefig(path.join(report_dir, f"{source}-mr-art-confusion"))

    recap = pd.DataFrame(
        base_metrics,
        columns=["source", "balanced_accuracy", "f1_0", "f1_1", "f1_2"],
    )
    recap.to_csv(path.join(report_dir, "mrart_recap.csv"))

    dropout_df.to_csv(path.join(report_dir, "dropout.csv"))
    simple_df.to_csv(path.join(report_dir, "test-pred.csv"))
    conf_df.to_csv(path.join(report_dir, "confidence.csv"))
    confidence_fig.savefig(path.join(report_dir, "confidence"))
    filtered_fig.savefig(path.join(report_dir, "filtered"))
    plt.close()


def test_transfer_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        print(f"Start Evaluation for {ckpt}")
        exp = path.basename(ckpt).split(".")[0]
        report_dir = path.join("test_report", "transfer", exp)
        if path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        module = MrArtTransferTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(module=module, report_dir=report_dir)


def test_unbalanced_transfer_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        print(f"Start Evaluation for {ckpt}")
        exp = path.basename(ckpt).split(".")[0]
        report_dir = path.join("test_report", "unbalanced", "transfer", exp)
        if path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        module = MrArtTransferTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(
            module=module, report_dir=report_dir, dataset_class=TestUnbalancedMrArt
        )


def test_unbalanced_scratch_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        print(f"Start Evaluation for {ckpt}")
        exp = path.basename(ckpt).split(".")[0]
        report_dir = path.join("test_report", "unbalanced", "scratch", exp)
        if path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        module = MRArtScratchTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(
            module=module, report_dir=report_dir, dataset_class=TestUnbalancedMrArt
        )


def test_unbalanced_pretrain_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        print(f"Start Evaluation for {ckpt}")
        exp = path.basename(ckpt).split(".")[0]
        report_dir = path.join("test_report", "unbalanced", "pretrain", exp)
        if path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        module, task = task_utils.load_pretrain_from_ckpt(ckpt_path=ckpt)

        test_pretrain_model_mrart_data(
            module=module,
            report_dir=report_dir,
            datasets=[("train", TrainUnbalancedMrArt), ("test", TestUnbalancedMrArt)],
            prefix="unbalanced-mrart",
        )
