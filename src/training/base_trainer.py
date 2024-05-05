import logging
import tempfile
import random
import sys

sys.path.append(".")
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset

import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from comet_ml.integration.pytorch import log_model

from src.network.res_net import ResNetModel
from src.network.base_net import BaselineModel
from src.dataset.mrart_dataset import TrainMrArt, ValMrArt
from src.transforms.base_transform import Preprocess, Augment
from src.network.utils import init_weights
from src.training.arg_parser import create_arg_parser



IM_SHAPE = (1, 160, 192, 160)


def launch_train(config):
    
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name="test_soft_encode",
        experiment_name=f"Soft encode",
    )
    comet_logger.log_hyperparams({"seed":config.seed})
    train_tsf = Preprocess(soft_labeling=True)
    train_ds = (
        TrainMrArt.narval(train_tsf)
        if config.narval
        else TrainMrArt.lab(train_tsf)
    )
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
        ValMrArt.narval(train_tsf) if config.narval else ValMrArt.lab(train_tsf),
        batch_size=config.batch_size,
    )
    logging.info(f"Dataset contain {len(train_ds)} datas")
    tempdir = tempfile.TemporaryDirectory()

    if config.model =="BASE":
        net = BaselineModel(
            1,
            IM_SHAPE,
            act=config.act,
            kernel_size=config.conv_k,
            run_name=tempdir.name,
            lr=config.learning_rate,
            beta=config.beta,
            use_decoder=config.use_decoder,
        )
    elif config.model == "RES":
            net = ResNetModel(
            1,
            IM_SHAPE,
            act=config.act,
            kernel_size=config.conv_k,
            run_name=tempdir.name,
            lr=config.learning_rate,
            beta=config.beta,
            use_decoder=config.use_decoder,
        )

    net.apply(init_weights)

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

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log_model(
        comet_logger.experiment,
        BaselineModel.load_from_checkpoint(check.best_model_path),
        "BASE",
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    config = create_arg_parser()
    logging.info(str(config))
    if config.seed == None:
        config.seed = random.randint(1, 10000)
    torch.manual_seed(config.seed)

    torch.set_float32_matmul_precision("high")
    launch_train(config)
