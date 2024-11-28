import json
import os
import shutil
from functools import partial
from typing import Any

import lightning as L
import torch
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from src import config as conf
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.network.archi import Model
from src.training.scratch_logic import AMPSCZScratchTask
from src.training.transfer_logic import AMPSCZTransferTask
from src.utils.task import load_pretrain_from_ckpt


def scratch_tune(
    config: dict[str, Any],
):
    torch.cuda.empty_cache()
    os.chdir(
        "/home/cbricout/projects/ctb-sbouix/cbricout/mrart"
        if conf.IS_NARVAL
        else "/home/at70870/Desktop/mrart"
    )
    model = AMPSCZScratchTask(
        conf.IM_SHAPE,
        lr=config["lr"],
        dropout_rate=config["dropout_rate"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"],
    )
    trainer = L.Trainer(
        devices="auto",
        accelerator="gpu",
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
        default_root_dir=os.environ.get("SLURM_TMPDIR", "temp/"),
    )
    datamodule = AMPSCZDataModule(config["batch_size"])
    trainer.fit(model, datamodule=datamodule)


def run_scratch_tune():
    report_dir = f"hp_tune/scratch/SFCN"
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)
    os.environ["TQDM_DISABLE"] = "True"
    config = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([12, 14, 16]),
        "dropout_rate": tune.uniform(0.6, 0.8),
        "weight_decay": tune.uniform(0.01, 0.2),
    }

    scheduler = ASHAScheduler(
        max_t=60,
        grace_period=20,
        reduction_factor=2,
    )

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 5, "GPU": 1},
        trainer_resources={"CPU": 1, "GPU": 0},
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_balanced_accuracy",
            checkpoint_score_order="max",
        ),
        storage_path=conf.RAYTUNE_DIR,
        name="SFCN",
        progress_reporter=CLIReporter(metric_columns=["val_balanced_accuracy"]),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        scratch_tune,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="val_balanced_accuracy",
            mode="max",
            num_samples=250,
            scheduler=scheduler,
        ),
    )
    result_grid = tuner.fit()

    best_res = result_grid.get_best_result(
        metric="val_balanced_accuracy", mode="max", scope="avg"
    )
    best_dict = best_res.config["train_loop_config"]
    best_dict["val_balanced_accuracy_mean"] = best_res.metrics_dataframe[
        "val_balanced_accuracy"
    ].mean()
    best_dict["val_balanced_accuracy_max"] = best_res.metrics_dataframe[
        "val_balanced_accuracy"
    ].max()
    with open(f"{report_dir}/best.json", "w") as convert_file:
        convert_file.write(json.dumps(best_dict))

    print(f"Best result with {best_res.config}")
    df = result_grid.get_dataframe(
        filter_metric="val_balanced_accuracy", filter_mode="max"
    )
    df.to_csv(f"{report_dir}/grid_search_scratch.csv")


def transfer_tune(
    config: dict[str, Any],
    pretrained: Model,
):
    torch.cuda.empty_cache()
    os.chdir(
        "/home/cbricout/projects/ctb-sbouix/cbricout/mrart"
        if conf.IS_NARVAL
        else "/home/at70870/Desktop/mrart"
    )
    model = AMPSCZTransferTask(
        input_size=pretrained.encoder.latent_shape,
        pretrained=pretrained,
        lr=config["lr"],
        dropout_rate=config["dropout_rate"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"],
        num_layers=config["num_layers"],
    )
    trainer = L.Trainer(
        devices="auto",
        accelerator="gpu",
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
        default_root_dir=os.environ.get("SLURM_TMPDIR", "temp/"),
    )
    datamodule = AMPSCZDataModule(
        config["batch_size"], pretrained_model=pretrained.encoder
    )
    trainer.fit(model, datamodule=datamodule)


def run_transfer_tune(pretrain_path: str):
    print(f"tuning for pretrained model {pretrain_path}")
    report_dir = f"hp_tune/transfer/{os.path.basename(pretrain_path)}".replace(
        ".ckpt", ""
    )
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)
    os.environ["TQDM_DISABLE"] = "True"
    config = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([12, 14, 16]),
        "dropout_rate": tune.uniform(0.6, 0.8),
        "weight_decay": tune.uniform(0.01, 0.2),
        "num_layers": tune.choice([0, 1, 2, 3]),
    }

    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=20,
        reduction_factor=2,
    )

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 0.25},
        trainer_resources={"CPU": 1, "GPU": 0},
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="val_balanced_accuracy",
            checkpoint_score_order="max",
        ),
        progress_reporter=CLIReporter(metric_columns=["val_balanced_accuracy"]),
        storage_path=conf.RAYTUNE_DIR,
        name=os.path.basename(pretrain_path).replace(".ckpt", ""),
    )

    pretrained = load_pretrain_from_ckpt(pretrain_path)
    model = pretrained.model.float()
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        partial(transfer_tune, pretrained=model),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="val_balanced_accuracy",
            mode="max",
            num_samples=500,
            scheduler=scheduler,
        ),
    )
    result_grid = tuner.fit()

    best_res = result_grid.get_best_result(
        metric="val_balanced_accuracy", mode="max", scope="avg"
    )
    best_dict = best_res.config["train_loop_config"]
    best_dict["val_balanced_accuracy_mean"] = best_res.metrics_dataframe[
        "val_balanced_accuracy"
    ].mean()
    best_dict["val_balanced_accuracy_max"] = best_res.metrics_dataframe[
        "val_balanced_accuracy"
    ].max()
    with open(f"{report_dir}/best.json", "w") as convert_file:
        convert_file.write(json.dumps(best_dict))

    print(f"Best result with {best_res.config}")
    df = result_grid.get_dataframe(
        filter_metric="val_balanced_accuracy", filter_mode="max"
    )
    df.to_csv(f"{report_dir}/grid_search_transfer.csv")
