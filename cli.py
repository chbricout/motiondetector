import logging
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
from src.commands.pretrainer import launch_pretrain

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
    default=0.2,
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
    default="MRART",
    type=str,
)
model = click.option(
    "--model",
    help="Model architecture : CNN, RES, SFCN, CONV5_FC3, SERES, VIT",
    default="CNN",
    type=str,
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
narval = click.option(
    "-n",
    "--narval",
    help="Flag if running on narval (for datasets paths)",
    is_flag=True,
    type=bool,
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
@narval
@slurm
@cutout
def pretrain(
    max_epochs,
    learning_rate,
    dropout_rate,
    batch_size,
    model,
    run_num,
    seed,
    narval,
    slurm,
    cutout,
):
    if slurm:
        submit_pretrain(
            model=model,
            array=run_num,
        )
    else:
        logging.basicConfig(level="INFO")
        launch_pretrain(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
            narval=narval,
            use_cutout=cutout,
        )


@cli.command()
@max_epoch
@learning_rate
@dataset
@batch_size
@model
@run_num
@seed
@narval
@slurm
def finetune(
    max_epochs, learning_rate, dataset, batch_size, model, run_num, seed, narval, slurm
):
    if slurm:
        submit_finetune(
            model=model,
            array=run_num,
        )
    else:
        logging.basicConfig(level="INFO")
        launch_finetune(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
            narval=narval,
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
@narval
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
    narval,
    slurm,
):
    if slurm:
        submit_scratch(
            model=model,
            array=run_num,
        )
    else:
        logging.basicConfig(level="INFO")
        launch_train_from_scratch(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            run_num=run_num,
            seed=seed,
            narval=narval,
        )


@cli.command()
@click.option(
    "-n",
    "--new_dataset",
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
def pretrainer(cutout: bool, test: bool):
    for model in run_confs:

        cmd = f"cli.py pretrain -n  \
                --batch_size {model['batch_size']}\
                --model {model['name']}\
                --learning_rate 8e-5\
                --dropout_rate 0.75 "
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
                f"cli.py finetune -n  \
                    --batch_size {model['batch_size']}\
                    --model {model['name']}\
                    --learning_rate 1e-5\
                    --dataset {dataset} ",
                dataset=dataset,
            )


if __name__ == "__main__":
    cli()
