import tempfile
import random
import sys

sys.path.append(".")
import comet_ml
import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.dataset.mrart.mrart_dataset import MRArtDataModule
from src.dataset.ampscz.ampscz_dataset import AMPSCZDataModule
from src.config import IM_SHAPE, PROJECT_NAME
from src.training.callback import FinetuneCallback
from src.training.lightning_logic import (
    MRArtFinetuningTask,
    AMPSCZFinetuningTask,
    PretrainingTask,
)


def launch_finetune(
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    dataset: str,
    model: str,
    run_num: int,
    seed: int,
    narval: bool,
):
    torch.set_float32_matmul_precision("high")

    if dataset == "MRART":
        datamodule = MRArtDataModule
        task = MRArtFinetuningTask
    elif dataset == "AMPSCZ":
        datamodule = AMPSCZDataModule
        task = AMPSCZFinetuningTask

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name=PROJECT_NAME,
        experiment_name=f"finetune-{dataset}-{model}-{run_num}",
    )

    if seed == None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams({"seed": seed})

    api = comet_ml.api.API(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
    )

    tempdir = tempfile.TemporaryDirectory()

    pretrain_exp = api.get("mrart", PROJECT_NAME, f"pretraining-{model}-{run_num}")
    pretrain_exp.download_model(
        model,
        output_path=f"/home/cbricout/scratch/{PROJECT_NAME}-{run_num}/{model}",
    )
    pretrained = PretrainingTask.load_from_checkpoint(
        f"/home/cbricout/scratch/{PROJECT_NAME}-{run_num}/{model}/model-data/comet-torch-model.pth"
    )
    net = task(pretrained_model=pretrained.model, im_shape=IM_SHAPE, lr=learning_rate)

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=100),
            FinetuneCallback(monitor="val_balanced_accuracy", mode="max"),
        ],
    )

    trainer.fit(net, datamodule=datamodule(narval, batch_size))
