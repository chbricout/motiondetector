import logging
import tempfile
import random
import sys
import os

from src.utils.comet import get_experiment_key

sys.path.append(".")
import torch
import lightning
import lightning.pytorch.loggers
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping,LearningRateMonitor
from src.dataset.pretraining.pretraining_dataset import PretrainingDataModule
from src.training.callback import PretrainCallback
from src.training.lightning_logic import PretrainingTask
from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME


def launch_pretrain(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    model: str,
    run_num: int,
    seed: int,
    narval: bool,
):
    run_name = f"pretraining-{model}-{run_num}"
    if not os.path.exists(f"/home/cbricout/scratch/{PROJECT_NAME}"):
        os.mkdir(f"/home/cbricout/scratch/{PROJECT_NAME}")
    run_dir = f"/home/cbricout/scratch/{PROJECT_NAME}/{run_name}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    comet_logger = lightning.pytorch.loggers.CometLogger(
        api_key=COMET_API_KEY,
        project_name=PROJECT_NAME,
        experiment_name=run_name,
        experiment_key=get_experiment_key(
            COMET_API_KEY, "mrart", PROJECT_NAME, run_name
        ),
    )

    if seed == None:
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    comet_logger.log_hyperparams({"seed": seed})
    comet_logger.experiment.log_code(file_name="src/commands/pretrainer.py")
    logging.info(f"Run dir path is : {run_dir}")
    net = PretrainingTask(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        devices=2,
        strategy="ddp",
        accelerator="gpu",
        precision="16-mixed",
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(
                monitor="r2_score",
                mode="max",
                patience=20,
                check_on_train_epoch_end=False,
                verbose=True
            ),
            PretrainCallback(monitor="r2_score", mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
            # StochasticWeightAveraging(swa_lrs=learning_rate*0.2, swa_epoch_start=70)
        ],
    )


    trainer.fit(net, datamodule=PretrainingDataModule(narval, batch_size))
