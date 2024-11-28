import glob
import os
import shutil
from os import path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

import src.training.eval as teval
from src.dataset.ampscz.ampscz_dataset import FullTestAMPSCZ
from src.dataset.base_dataset import BaseDataset
from src.dataset.pretraining.pretraining_dataset import PretrainTest
from src.training.common_logic import BaseFinalTrain, get_calibration_curve
from src.training.pretrain_logic import PretrainingTask
from src.training.scratch_logic import AMPSCZScratchTask
from src.training.transfer_logic import AMPSCZTransferTask
from src.transforms.load import FinetuneTransform, LoadSynth
from src.utils import metrics
from src.utils import task as task_utils


def test_pretrain_in_folder(folder: str):
    models_ckpt = glob.glob(path.join(folder, "*.ckpt"))

    for ckpt in models_ckpt:
        module, report_dir = setup_test_pretrain(ckpt)
        test_pretrain_model_pretrain_data(module=module, report_dir=report_dir)


def setup_test_pretrain(ckpt_path: str) -> tuple[PretrainingTask, str, str]:
    module = task_utils.load_pretrain_from_ckpt(ckpt_path=ckpt_path)
    print(f"Start Evaluation for {ckpt_path}")

    exp = path.basename(ckpt_path).split(".")[0]
    report_dir = path.join("test_report", "pretraining", exp)
    if path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    return module, report_dir


def test_pretrain_model_pretrain_data(module: PretrainingTask, report_dir: str):

    ds = PretrainTest.from_env(LoadSynth.from_task())
    dataloader = DataLoader(
        ds,
        batch_size=24,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=2,
    )
    print("Predict pretrain")
    simple_df = teval.get_pred_from_pretrain(module, dataloader, label="motion_mm")
    print("Pretrain DONE")

    base_metrics = [
        r2_score(simple_df["label"], simple_df["pred"]),
        mean_squared_error(simple_df["label"], simple_df["pred"], squared=False),
    ]

    base_metrics_df = pd.DataFrame(
        base_metrics,
        columns=["r2", "rmse"],
    )
    base_metrics_df.to_csv(path.join(report_dir, "results.csv"))
    calib_fig = get_calibration_curve(simple_df["pred"], simple_df["label"])
    calib_fig.savefig(path.join(report_dir, "calibration"))
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
        module = AMPSCZScratchTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(
            module=module, report_dir=report_dir, dataset_class=FullTestAMPSCZ
        )


def test_scratch_model(
    module: BaseFinalTrain, report_dir: str, dataset_class: BaseDataset
):
    dl = DataLoader(dataset_class.from_env(FinetuneTransform()))
    simple_df = teval.get_pred_from_pretrain(module, dl)

    acc, per_class_f1, cm_fig = metrics.prediction_report(
        simple_df["label"], simple_df["pred"].astype(int)
    )
    base_metrics = [acc, *per_class_f1]
    cm_fig.tight_layout()
    cm_fig.savefig(path.join(report_dir, f"confusion"))

    recap = pd.DataFrame(
        base_metrics,
        columns=["balanced_accuracy", "f1_0", "f1_1", "f1_2"],
    )
    recap.to_csv(path.join(report_dir, "recap.csv"))

    simple_df.to_csv(path.join(report_dir, "test-pred.csv"))
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
        module = AMPSCZTransferTask.load_from_checkpoint(checkpoint_path=ckpt)
        test_scratch_model(
            module=module, report_dir=report_dir, dataset_class=FullTestAMPSCZ
        )
