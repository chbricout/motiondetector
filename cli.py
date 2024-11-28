import json
import os
import warnings
from glob import glob

import click

from src import config
from src.commands.base_trainer import launch_train_from_scratch
from src.commands.generate_datasets import launch_generate_data
from src.commands.launch_slurm import (
    submit_generate_ds,
    submit_pretrain,
    submit_scratch,
    submit_transfer,
    submit_tune_scratch,
    submit_tune_transfer,
)
from src.commands.pretrainer import launch_pretrain
from src.commands.test_models import (
    test_pretrain_in_folder,
    test_scratch_in_folder,
    test_scratch_model,
    test_transfer_in_folder,
)
from src.commands.transfer import launch_transfer
from src.commands.tune import run_scratch_tune, run_transfer_tune
from src.utils.log import lightning_logger, rich_logger

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cuda.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cudnn.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mkldnn.*")


max_epoch = click.option(
    "--max_epochs",
    help="Max epochs to train",
    default=1000,
    type=int,
)
learning_rate = click.option(
    "--learning_rate",
    help="learning rate",
    default=1e-5,
    type=float,
)
dropout_rate = click.option(
    "--dropout_rate",
    help="dropout rate",
    default=0.7,
    type=float,
)
weight_decay = click.option(
    "--weight_decay",
    help="AdamW weight decay",
    default=0.05,
    type=float,
)
batch_size = click.option(
    "--batch_size",
    help="Batch size for training",
    default=12,
    type=int,
)

run_num = click.option(
    "--run_num",
    help="Identifier of job in the array job list",
    default=1,
    type=int,
)
seed = click.option(
    "--seed",
    help="Random seed for torch",
    default=None,
    type=int,
)
slurm = click.option(
    "-S",
    "--slurm",
    help="Flag to submit corresponding slurm job",
    is_flag=True,
    type=bool,
)
account = click.option(
    "-A",
    "--account",
    help="Slurm accoun",
    default=config.DEFAULT_SLURM_ACCOUNT,
    type=click.Choice(
        ["ctb-sbouix", "def-sbouix", "rrg-ebrahimi", "def-ebrahimi"],
        case_sensitive=True,
    ),
)


@click.group()
def cli():
    pass


@cli.command()
@max_epoch
@learning_rate
@dropout_rate
@batch_size
@run_num
@seed
@slurm
@account
def pretrain(
    max_epochs,
    learning_rate,
    dropout_rate,
    batch_size,
    run_num,
    seed,
    slurm,
    account,
):
    if slurm:
        submit_pretrain(array=run_num, account=account)
    else:
        lightning_logger()
        launch_pretrain(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            run_num=run_num,
            seed=seed,
        )


@cli.command()
@click.option(
    "-M",
    "--pretrain_path",
    help="Pretrain model to use",
    type=str,
)
@max_epoch
@learning_rate
@dropout_rate
@batch_size
@weight_decay
@click.option(
    "--num_layers",
    help="Number of layers in transfer network",
    type=int,
)
@run_num
@seed
@slurm
def transfer(
    pretrain_path,
    max_epochs,
    learning_rate,
    dropout_rate,
    batch_size,
    num_layers,
    weight_decay,
    run_num,
    seed,
    slurm,
):
    if slurm:
        submit_transfer(
            pretrain_path=pretrain_path,
            array=range(1, 6),
        )
    else:
        lightning_logger()
        launch_transfer(
            pretrain_path=pretrain_path,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            run_num=run_num,
            num_layers=num_layers,
            weight_decay=weight_decay,
            seed=seed,
        )


@cli.command()
@max_epoch
@learning_rate
@dropout_rate
@batch_size
@weight_decay
@run_num
@seed
@slurm
def train(
    max_epochs,
    learning_rate,
    dropout_rate,
    batch_size,
    weight_decay,
    run_num,
    seed,
    slurm,
):
    if slurm:
        submit_scratch(array=run_num)
    else:
        lightning_logger()
        launch_train_from_scratch(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            run_num=run_num,
            seed=seed,
        )


@cli.command()
@click.option(
    "-d",
    "--new_dataset",
    help="New dataset name",
    default="pretraining-motion",
    type=str,
)
@slurm
def generate_data(new_dataset, slurm: bool):
    rich_logger()
    if slurm:
        submit_generate_ds()
    else:
        launch_generate_data(new_dataset)


@cli.group()
def launch_exp():
    pass


run_confs = [
    {"name": "VIT", "batch_size": 18},
    {"name": "SFCN", "batch_size": 24},
    {"name": "CONV5_FC3", "batch_size": 24},
    {"name": "RES", "batch_size": 24},
    {"name": "SERES", "batch_size": 24},
]


@launch_exp.command()
@click.option(
    "-t",
    "--test",
    help="Flag to run one run for each conf (no array job)",
    is_flag=True,
    type=bool,
)
def pretrainer(test: bool):
    for model in run_confs:

        cmd = f"cli.py pretrain  \
                --batch_size {model['batch_size']}\
                --model {model['name']}\
                --learning_rate 2e-5\
                --dropout_rate 0.6"

        array = range(1, 6)
        if test:
            array = 1

        submit_pretrain(
            array,
            cmd,
        )


@launch_exp.command()
@click.option(
    "-d",
    "--directory",
    help="Directory with pretrained models",
    type=str,
)
def transfer(directory: str):
    with open("run_config.json", "r") as file:
        setting = json.load(file)["transfer"]
        for model in glob(os.path.join(directory, "*.ckpt")):
            model_name = os.path.basename(model).removesuffix(".ckpt")
            conf = setting[model_name]
            submit_transfer(
                model,
                range(1, 6),
                f"cli.py transfer   \
                    --batch_size {conf['batch_size']}\
                    --dropout_rate {conf['dropout_rate']}\
                    --weight_decay {conf['weight_decay']}\
                    --num_layers {conf['num_layers']}\
                    --learning_rate {conf['lr']}\
                    --pretrain_path {model}\
                    --max_epochs 100000",
            )


@launch_exp.command()
def train():
    with open("run_config.json", "r") as file:
        setting = json.load(file)["scratch"]
        for _, conf in setting.items():
            submit_scratch(
                list(range(1, 6)),
                f"cli.py train   \
                    --max_epochs 10000\
                    --batch_size {conf['batch_size']}\
                    --learning_rate {conf['lr']}\
                    --dropout_rate {conf['dropout_rate']}\
                    --weight_decay {conf['weight_decay']}",
            )


@cli.group()
def plot():
    pass


@cli.group()
def test():
    pass


@test.command("pretrain")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def pretrain_test(directory: str):
    test_pretrain_in_folder(directory)


@test.command("scratch")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
@click.option("-f", "--file", help="File containing model", type=str, default=None)
def scratch_test(directory: str, file: str):
    if file is not None:
        test_scratch_model(ckpt_path=file)
    else:
        test_scratch_in_folder(directory)


@test.command("transfer")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def transfer_test(directory: str):
    test_transfer_in_folder(directory)


@cli.group()
def tune():
    pass


@tune.command("scratch")
@click.option("-A", "--all", help="Use all models", type=bool, is_flag=True)
def tune_scratch(all: bool):
    if not all:
        run_scratch_tune()
    else:
        submit_tune_scratch(f"cli.py tune scratch")


@tune.command("transfer")
@click.option("-f", "--file", help="File containing model", type=str, default=None)
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def tune_transfer(file: str, directory: str):
    if not directory:
        run_transfer_tune(file)
    else:
        for model in glob(os.path.join(directory, "*.ckpt")):
            submit_tune_transfer(cmd=f"cli.py tune transfer --file {model}")


if __name__ == "__main__":
    cli()
