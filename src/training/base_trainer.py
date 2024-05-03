import logging
import sys
import tempfile

sys.path.append(".")
import comet_ml
from monai.data import CacheDataset, DataLoader, Dataset, ZipDataset
from monai.transforms import (
    RandFlip,
    RandRotate,
    Compose,
)
import torch
import torch.nn as nn
import lightning
from comet_ml.integration.pytorch import log_model

from src.network.base_net import BaselineModel
from src.dataset.mrart_dataset import TrainMrArt, ValMrArt
from src.transforms.base_transform import Preprocess, Augment
from src.network.utils import init_weights
from src.training.arg_parser import create_arg_parser

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

IM_SHAPE=(1,160,192,160)

def launch_train(config):
    name = "Base"
    if not config.use_decoder:
        name += "-NoDec"
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name="nan-investigate-midl2024",
        experiment_name=f"{name}-beta{config.beta}-lr{config.learning_rate}",
    )

    train_ds = TrainMrArt.lab(Preprocess())
    flipped_train_ds = Dataset(
        data=train_ds,
        transform=Augment(),
    )

    train_loader = DataLoader(
        flipped_train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        ValMrArt.lab(Preprocess()), batch_size=config.batch_size
    )
    logging.info(f"Dataset contain {len(train_ds)} datas")
    tempdir = tempfile.TemporaryDirectory()
    aug_net = BaselineModel(
        1,
        IM_SHAPE,
        act=config.act,
        kernel_size=config.conv_k,
        run_name=tempdir.name,
        lr=config.learning_rate,
        beta=config.beta,
        use_decoder=config.use_decoder,
    )

    aug_net.apply(init_weights)

    check = ModelCheckpoint(monitor="val_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            # EarlyStopping(monitor="val_label_loss", mode="min", patience=50),
            check,
        ],
    )
   
    trainer.fit(aug_net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log_model(
        comet_logger.experiment,
        BaselineModel.load_from_checkpoint(check.best_model_path),
        name,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    config = create_arg_parser()
    logging.info(str(config))
    torch.set_float32_matmul_precision("high")
    launch_train(config)
