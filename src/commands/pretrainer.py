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

def get_run_dir(project_name:str, run_name:str, narval:bool):
    if narval:
        root_dir = f"/home/cbricout/scratch/{PROJECT_NAME}"
    else :
        root_dir = f"/home/at70870/local_scratch/{PROJECT_NAME}"

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    run_dir = f"{root_dir}/{run_name}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    return run_dir

def launch_pretrain(
    learning_rate: float,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    model: str,
    run_num: int,
    seed: int,
    narval: bool,
    use_cutout:bool
):
    run_name = f"pretraining-{model}-{run_num}"
    run_dir = get_run_dir(PROJECT_NAME, run_name, narval)

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
    comet_logger.log_hyperparams({
        "seed": seed,
        "model":model,
        "run_num":run_num,
        "use_cutout":use_cutout
        })
    comet_logger.experiment.log_code(file_name="src/commands/pretrainer.py")
    logging.info(f"Run dir path is : {run_dir}")
    net = PretrainingTask(
        model_class=model,
        im_shape=IM_SHAPE,
        lr=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        use_cutout=use_cutout
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
                monitor="val_loss",
                mode="min",
                patience=40,
                check_on_train_epoch_end=False,
                verbose=True
            ),
            PretrainCallback(monitor="r2_score", mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )


    trainer.fit(net, datamodule=PretrainingDataModule(narval, batch_size))
