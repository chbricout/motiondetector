import logging
import tempfile
import random
import sys


sys.path.append(".")
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose

import comet_ml
import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from comet_ml.integration.pytorch import log_model

from src.network.res_net import ResNetModel
from src.network.base_net import BaselineModel
from src.network.SFCN import SFCNModel
from src.network.Conv5_FC3 import Conv5_FC3
from src.network.SENet import SEResModel

from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt
from src.dataset.ampscz.ampscz_dataset import FinetuneTrainAMPSCZ, FinetuneValAMPSCZ

from src.transforms.base_transform import Preprocess, FinalCrop
from src.training.arg_parser import create_arg_parser


IM_SHAPE = (1, 160, 192, 160)


def launch_train(config):

    if config.dataset == "MRART":
        ds_train_class = TrainMrArt
        ds_val_class = ValMrArt
        num_output = 3
    elif config.dataset == "AMPSCZ":
        ds_train_class = FinetuneTrainAMPSCZ
        ds_val_class = FinetuneValAMPSCZ
        num_output = 1
    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
        project_name="base-study",
        experiment_name=f"finetune-{config.dataset}-{config.model}-{'FREEZE' if config.freeze_encoder else ''}",
    )
    comet_logger.log_hyperparams({"seed": config.seed})

    api = comet_ml.api.API(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
    )

    preproc_tsf = Preprocess(soft_labeling=False)
    crop_tsf = FinalCrop()
    load_tsf = Compose([preproc_tsf, crop_tsf])

    train_mrart_ds = (
        ds_train_class.narval(load_tsf)
        if config.narval
        else ds_train_class.lab(load_tsf)
    )
    train_loader = DataLoader(
        train_mrart_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )

    val_mrart_ds = (
        ds_val_class.narval(load_tsf) if config.narval else ds_val_class.lab(load_tsf)
    )
    val_loader = DataLoader(
        val_mrart_ds,
        batch_size=config.batch_size,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info(f"Dataset contain {len(train_mrart_ds)} datas")
    tempdir = tempfile.TemporaryDirectory()
    model_class = None
    if config.model == "BASE":
        model_class = BaselineModel
        net = BaselineModel(
            1,
            IM_SHAPE,
            act=config.act,
            kernel_size=config.conv_k,
            run_name=tempdir.name,
            lr=config.learning_rate,
            beta=config.beta,
            use_decoder=config.use_decoder,
            mode=config.mode,
            dropout_rate=config.dropout_rate,
        )
    elif config.model == "RES":
        model_class = ResNetModel
        net = ResNetModel(
            1,
            IM_SHAPE,
            run_name=tempdir.name,
            lr=config.learning_rate,
            mode=config.mode,
        )
    elif config.model == "SFCN":
        model_class = SFCNModel
        net = SFCNModel(
            1,
            IM_SHAPE,
            run_name=tempdir.name,
            lr=config.learning_rate,
        )
    elif config.model == "Conv5_FC3":
        model_class = Conv5_FC3
        net = Conv5_FC3(
            1,
            IM_SHAPE,
            run_name=tempdir.name,
            lr=config.learning_rate,
        )
    elif config.model == "SERes":
        model_class = SEResModel
        net = SEResModel(
            1,
            IM_SHAPE,
            run_name=tempdir.name,
            lr=config.learning_rate,
            mode=config.mode,
        )
    pretrain_exp = api.get("mrart", "base-study", f"pretraining-{config.model}")
    pretrain_exp.download_model(config.model, output_path=config.model)
    net.load_state_dict(torch.load(f"{config.model}/model-data/comet-torch-model.pth"))
    ## Freeze model
    if config.freeze_encoder:
        net.encoder.requires_grad_(False)
    if num_output != 1:
        net.change_output_num(num_output)
    check = ModelCheckpoint(monitor="val_balanced_accuracy", mode="max")

    trainer = lightning.Trainer(
        max_epochs=config.max_epochs,
        logger=comet_logger,
        devices=[0],
        accelerator="gpu",
        default_root_dir=tempdir.name,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(monitor="val_balanced_accuracy", mode="max", patience=50),
            check,
        ],
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log_model(
        comet_logger.experiment,
        model_class.load_from_checkpoint(
            check.best_model_path, output_class=num_output
        ),
        config.model,
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
