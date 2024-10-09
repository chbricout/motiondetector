from glob import glob
import os
import warnings
import click

from src.commands.base_trainer import launch_train_from_scratch
from src.commands.test_models import (
    test_pretrain_in_folder,
    test_pretrain_model_mrart_data,
    test_pretrain_model_pretrain_data,
    test_scratch_in_folder,
    test_scratch_model,
    test_transfer_in_folder,
    test_unbalanced_pretrain_in_folder,
    test_unbalanced_scratch_in_folder,
    test_unbalanced_transfer_in_folder
)
from src.commands.transfer import launch_transfer
from src.commands.generate_datasets import launch_generate_data
from src.commands.launch_slurm import (
    submit_test_pretrain,
    submit_transfer,
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
    help="Dataset for finetuning mode : MRART, UNBALANCED-MRART or AMPSCZ",
    default="MRART",
    type=click.Choice(["MRART", "AMPSCZ", "UNBALANCED-MRART"], case_sensitive=True),
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
@model
@run_num
@seed
@slurm
@cutout
@task
@account
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
    account,
):
    if slurm:
        submit_pretrain(model=model, array=run_num, account=account)
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
@click.option(
    "-M",
    "--pretrain_path",
    help="Pretrain model to use",
    type=str,
)
@max_epoch
@learning_rate
@dataset
@batch_size
@run_num
@seed
@slurm
def transfer(
    pretrain_path, max_epochs, learning_rate, dataset, batch_size, run_num, seed, slurm
):
    if slurm:
        submit_transfer(
            pretrain_path=pretrain_path,
            array=range(1,6),
        )
    else:
        lightning_logger()
        launch_transfer(
            pretrain_path=pretrain_path,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dataset=dataset,
            batch_size=batch_size,
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
    {"name": "VIT", "batch_size": 18},
    {"name": "SFCN", "batch_size": 24},
    {"name": "CONV5_FC3", "batch_size": 24},
    {"name": "RES", "batch_size": 24},
    {"name": "SERES", "batch_size": 24},
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
                --dropout_rate 0.6\
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


transfer_confs = [
    {"name": "VIT", "batch_size": 16},
    {"name": "SFCN", "batch_size": 16},
    {"name": "CONV5_FC3", "batch_size": 16},
    {"name": "RES", "batch_size": 16},
    {"name": "SERES", "batch_size": 16},
]


@launch_exp.command()
@click.option(
    "-d",
    "--directory",
    help="Directory with pretrained models",
    type=str,
)
@dataset
def transfer(directory:str, dataset:str):
    for model in glob(os.path.join(directory, '*.ckpt')):
        submit_transfer(
            model,
            range(1,6),
            f"cli.py transfer   \
                --batch_size 24\
                --pretrain_path {model}\
                --learning_rate 1e-3\
                --dataset {dataset} \
                --max_epochs 100000",
            dataset=dataset,
        )
           

@launch_exp.command()
@dataset
def train(dataset:str):
    for model in transfer_confs:
        submit_scratch(
            model["name"],
            list(range(1,6)),
            f"cli.py train   \
                --max_epochs 10000\
                --batch_size {model['batch_size']}\
                --model {model['name']}\
                --learning_rate 1e-5\
                --dropout_rate 0.6\
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


@cli.group()
def test():
    pass


@test.command("pretrain")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
@click.option("-f", "--file", help="File containing model", type=str, default=None)
@slurm
def pretrain_test(directory: str, file: str, slurm: bool):
    if slurm:
        submit_test_pretrain(directory)
    else:
        if file is not None:
            test_pretrain_model_pretrain_data(file)
            test_pretrain_model_mrart_data(ckpt_path=file)
        else:
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

@test.command("transfer-unbalanced")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def transfer_unbalanced_test(directory: str):
    test_unbalanced_transfer_in_folder(directory)

@test.command("scratch-unbalanced")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def scratch_unbalanced_test(directory: str):
    test_unbalanced_scratch_in_folder(directory)

@test.command("pretrain-unbalanced")
@click.option(
    "-d", "--directory", help="Directory containing models", type=str, default=None
)
def pretrain_unbalanced_test(directory: str):
    test_unbalanced_pretrain_in_folder(directory)




if __name__ == "__main__":
    cli()
