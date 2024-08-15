import warnings
import click

from src.commands.base_trainer import launch_train_from_scratch
from src.commands.finetune import launch_finetune
from src.commands.generate_datasets import launch_generate_data
from src.commands.launch_slurm import (
    submit_finetune,
    submit_generate_ds,
    submit_pretrain,
    submit_scratch,
)
from src.commands.mr_art_to_bids import launch_convert_mrart_to_bids
from src.commands.plot import pretrain_calibration_gif
from src.commands.pretrainer import launch_pretrain
from src.utils.comet import export_torchscript
from src.utils.log import lightning_logger, rich_logger
from src import config

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
batch_size = click.option(
    "--batch_size",
    help="Batch size for training",
    default=12,
    type=int,
)
dataset = click.option(
    "--dataset",
    help="Dataset for finetuning mode : MRART or AMPSCZ",
    default=click.Choice(["MRART", "AMPSCZ"], case_sensitive=True),
    type=str,
)
model = click.option(
    "--model",
    help="Model architecture : CNN, RES, SFCN, CONV5_FC3, SERES, VIT",
    default="CNN",
    type=click.Choice(
        ["CNN", "RES", "SFCN", "CONV5_FC3", "SERES", "VIT"], case_sensitive=True
    ),
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
cutout = click.option(
    "--cutout",
    help="Flag to use cutout strategy in training",
    is_flag=True,
    type=bool,
)
project = click.option(
    "--project",
    help="Comet project to use",
    default=config.PROJECT_NAME,
    type=str,
)
task = click.option(
    "--task",
    help="Pretraining task : MOTION, SSIM, BINARY",
    default="MOTION",
    type=click.Choice(["MOTION", "SSIM", "BINARY"], case_sensitive=True),
)


@click.group()
def cli():
    pass


@cli.command()
@max_epoch
@learning_rate
@dropout_rate
@batch_size
@model
@run_num
@seed
@slurm
@cutout
@task
def pretrain(
    max_epochs,
    learning_rate,
    dropout_rate,
    batch_size,
    model,
    run_num,
    seed,
    slurm,
    cutout,
    task: str,
):
    if slurm:
        submit_pretrain(
            model=model,
            array=run_num,
        )
    else:
        lightning_logger()
        launch_pretrain(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
            use_cutout=cutout,
            task=task,
        )


@cli.command()
@max_epoch
@learning_rate
@dataset
@batch_size
@model
@run_num
@seed
@slurm
def finetune(
    max_epochs, learning_rate, dataset, batch_size, model, run_num, seed, slurm
):
    if slurm:
        submit_finetune(
            model=model,
            array=run_num,
        )
    else:
        lightning_logger()
        launch_finetune(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
        )


@cli.command()
@max_epoch
@learning_rate
@dropout_rate
@dataset
@batch_size
@model
@run_num
@seed
@slurm
def train(
    max_epochs,
    learning_rate,
    dropout_rate,
    dataset,
    batch_size,
    model,
    run_num,
    seed,
    slurm,
):
    if slurm:
        submit_scratch(model=model, array=run_num, dataset=dataset)
    else:
        lightning_logger()
        launch_train_from_scratch(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
        )


@cli.command()
@click.option(
    "-d",
    "-ew_dataset",
    help="New dataset name",
    default="pretraining-motion",
    type=str,
)
@slurm
def generate_data(new_dataset, slurm: bool):
    if slurm:
        submit_generate_ds()
    else:
        launch_generate_data(new_dataset)


@cli.command()
@click.option(
    "-i", "--input_path", help="Input path of MR-ART dataset", type=str, required=False
)
@click.option(
    "-o",
    "--output_path",
    help="Output path for BIDS converted MR-ART dataset",
    type=str,
    required=False,
)
def mrart_to_bids(input_path, output_path):
    launch_convert_mrart_to_bids(input_path, output_path)


@cli.command()
@model
@run_num
@project
@task
def compile_pretrain(model: str, run_num: int, project: str, task: str):
    rich_logger()
    export_torchscript(model, task, run_num, project_name=project)


@cli.group()
def launch_exp():
    pass


run_confs = [
    {"name": "VIT", "batch_size": 12},
    {"name": "SFCN", "batch_size": 30},
    {"name": "CNN", "batch_size": 30},
    {"name": "CONV5_FC3", "batch_size": 30},
    {"name": "RES", "batch_size": 30},
    {"name": "SERES", "batch_size": 30},
]


@launch_exp.command()
@cutout
@click.option(
    "-t",
    "--test",
    help="Flag to run one run for each conf (no array job)",
    is_flag=True,
    type=bool,
)
@task
def pretrainer(cutout: bool, test: bool, task: str):
    for model in run_confs:

        cmd = f"cli.py pretrain  \
                --batch_size {model['batch_size']}\
                --model {model['name']}\
                --learning_rate 2e-5\
                --dropout_rate 0.75\
                --task {task}"
        if cutout:
            cmd += " --cutout"

        array = range(1, 6)
        if test:
            array = 1

        submit_pretrain(
            model["name"],
            array,
            cmd,
        )


finetune_confs = [
    {"name": "VIT", "batch_size": 12},
    {"name": "SFCN", "batch_size": 28},
    {"name": "CNN", "batch_size": 28},
    {"name": "CONV5_FC3", "batch_size": 28},
]


@launch_exp.command()
def finetune():
    for model in finetune_confs:
        for dataset in ["MRART", "AMPSCZ"]:
            submit_finetune(
                model["name"],
                range(1, 6),
                f"cli.py finetune   \
                    --batch_size {model['batch_size']}\
                    --model {model['name']}\
                    --learning_rate 1e-5\
                    --dataset {dataset} ",
                dataset=dataset,
            )


@cli.group()
def plot():
    pass


@plot.command()
@model
@run_num
@project
@task
def calibration(model: str, run_num: int, project: str, task: str):
    rich_logger()
    pretrain_calibration_gif(
        model_name=model, task=task, run_num=run_num, project_name=project
    )


if __name__ == "__main__":
    cli()
