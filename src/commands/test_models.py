import glob
from os import path
import os
import shutil

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
import seaborn as sb

from src.dataset.mrart.mrart_dataset import TestMrArt, TrainMrArt
from src.dataset.pretraining.pretraining_dataset import (
    PretrainTest,
    PretrainVal,
    PretrainTrain,
)
from src.training.common_logic import BaseFinalTrain
from src.training.pretrain_logic import PretrainingTask
from src.transforms.load import FinetuneTransform, LoadSynth
from src.utils import metrics, task as task_utils
import src.training.eval as teval
from src.utils import mcdropout
from src.utils import confidence as conf


def test_pretrain_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        module, task, report_dir = setup_test_pretrain(ckpt)
        test_pretrain_model_pretrain_data(module=module, task=task, report_dir=report_dir)
        test_pretrain_model_mrart_data(module=module,  report_dir=report_dir)


def load_from_ckpt(ckpt_path: str):
    base_name = path.basename(ckpt_path)
    _, task, *_ = base_name.split("-")
    task_class = task_utils.str_to_task(task)
    return task_class.load_from_checkpoint(checkpoint_path=ckpt_path), task


def setup_test_pretrain(ckpt_path: str) -> tuple[PretrainingTask, str, str]:
    module, task = load_from_ckpt(ckpt_path=ckpt_path)
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
    dropout_pretrain_df = mcdropout.pretrain_mcdropout(
        module, dataloader, n_preds=1_000, label=label
    )
    print("MC-Dropout DONE")

    return simple_pretrain_df, dropout_pretrain_df


def test_pretrain_model_pretrain_data(module:PretrainingTask, task:str, report_dir:str):
    base_metrics = []

    ds = PretrainTest.from_env(LoadSynth.from_task(task))
    ds.define_label(task)

    simple_df, dropout_df = inference_test(module, ds, task_utils.label_from_task(task))

    for df, source, pred_label in [
        (simple_df, "simple", "pred"),
        (dropout_df, "mcdropout", "mean"),
    ]:
        base_metrics.append(
            [
                source,
                r2_score(df["label"], df[pred_label]),
                mean_squared_error(df["label"], df[pred_label], squared=False),
            ]
        )

    conf_df = conf.confidence_pretrain(dropout_df)

    fig = conf.plot_confidence(
        conf_df=conf_df,
        threshold_label="threshold_std",
        metric_label="rmse",
        threshold_axis="Threshold Standard Deviation",
        metric_axis="Root Mean Squared Error",
    )
    fig.savefig(path.join(report_dir, "test"))

    base_metrics_df = pd.DataFrame(base_metrics, columns=["source", "r2", "rmse"])
    base_metrics_df.to_csv(path.join(report_dir, "results.csv"))
    conf_df.to_csv(path.join(report_dir, "confidence.csv"))


def test_pretrain_model_mrart_data(module:PretrainingTask, report_dir:str):
    all_mode_df = []
    for mode, dataset in [("train", TrainMrArt), ("test", TestMrArt)]:
        dl = DataLoader(dataset.from_env(FinetuneTransform()))
        all_mode_df.append(teval.get_pred_from_pretrain(module, dl, mode))
    all_mode_df = pd.concat(all_mode_df)
    acc, per_class_accuracy, thresholds,fig_thresh, cm_fig = teval.separation_capacity(
        all_mode_df, train_on="train", test_on="test"
    )

    fig_thresh.tight_layout()
    fig_thresh.savefig(path.join(report_dir, "mr-art-thresholds"))

    cm_fig.tight_layout()
    cm_fig.savefig(path.join(report_dir, "mr-art-confusion"))

    all_mode_df.to_csv(path.join(report_dir, "mrart.csv"))

    recap=pd.DataFrame(
        [[acc, thresholds[0], thresholds[1], per_class_accuracy[0], per_class_accuracy[1], per_class_accuracy[2]]],
        columns=["balanced_accuracy", "threshold_1", "threshold_2", "accuracy_0", "accuracy_1", "accuracy_2"],
    )
    recap.to_csv(path.join(report_dir,"mrart_recap.csv"))

def test_mrart_model(module:BaseFinalTrain, report_dir:str):
    dl = DataLoader(TestMrArt.from_env(FinetuneTransform()))
    test_pred_df = teval.get_pred_from_pretrain(module, dl)
    
    acc, per_class_accuracy, cm_fig= metrics.prediction_report(test_pred_df['label'], test_pred_df['pred'])

    cm_fig.tight_layout()
    cm_fig.savefig(path.join(report_dir, "mr-art-confusion"))

    test_pred_df.to_csv(path.join(report_dir, "test-pred.csv"))

    recap=pd.DataFrame(
        [[acc,  per_class_accuracy[0], per_class_accuracy[1], per_class_accuracy[2]]],
        columns=["balanced_accuracy", "accuracy_0", "accuracy_1", "accuracy_2"],
    )
    recap.to_csv(path.join(report_dir,"mrart_recap.csv"))
