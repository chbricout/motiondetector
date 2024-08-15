"""Module defining comet utility functions"""

import glob
import logging
import os
import shutil
import tempfile
import comet_ml
from matplotlib.figure import Figure
import torch
from src import config
from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME
from src.network.utils import parse_model
from src.training.lightning_logic import PretrainingTask


def export_torchscript(
    model_name: str,
    task: str,
    run_num: int,
    project_name: str = PROJECT_NAME,
    api_key: str = COMET_API_KEY,
):
    """Export model from comet to torchscript, generate two files:
    - `{model_name}-{run_num}-mcdropout.pt` for Monte Carlo prediction
    - `{model_name}-{run_num}.pt` for normal prediction

    Args:
        model_name (str): name of the pretraining model to download.
        task (str): task of the pretraining model to download.
        run_num (int): specific run num to load.
        project_name (str): name of the comet project. Defaults to PROJECT_NAME.
        api_key (str, optional): comet api key. Defaults to COMET_API_KEY.
    """
    net = get_pretrain_task(
        model_name=model_name,
        task=task,
        run_num=run_num,
        project_name=project_name,
        api_key=api_key,
        del_folers=True,
    )

    export_dir = os.path.join("exports", project_name, model_name)
    mcdropout_file = os.path.join(export_dir, f"{model_name}-{run_num}-mcdropout.pt")
    eval_file = os.path.join(export_dir, f"{model_name}-{run_num}.pt")
    os.makedirs(export_dir)
    logging.info("Model retrieved ! \nTracing MC Dropout...")
    net.model.mc_dropout()
    torch.jit.trace(net.model, torch.rand(1, *IM_SHAPE).cuda()).save(mcdropout_file)

    logging.info("MCDropout saved (%s) ! \nTracing eval...", mcdropout_file)
    net.model.eval()
    torch.jit.trace(net.model, torch.rand(1, *IM_SHAPE).cuda()).save(eval_file)
    logging.info("Eval saved ! (%s)", eval_file)


def get_pretrain_task(
    model_name: str,
    task: str,
    run_num: int,
    project_name: str,
    api_key: str = COMET_API_KEY,
    del_folers: bool = False,
) -> PretrainingTask:
    """Retrieve pretrain task from comet, used for Finetuning

    Args:
        model_name (str): name of the model to download
        task (str): task of the model to download
        run_num (int): specific run num to load
        project_name (str): name of the comet project
        api_key (str, optional): comet api key. Defaults to COMET_API_KEY.
        del_folders (bool, optional): flag to delete exp folders after loading.
            Defaults to False

    Returns:
        PretrainingTask: loaded pretrained task
    """
    api = comet_ml.api.API(
        api_key=api_key,
    )
    pretrain_exp = api.get(
        "mrart", project_name, f"pretraining-{task}-{model_name}-{run_num}"
    )
    model_class = parse_model(model_name)

    if config.IS_NARVAL:
        output_dir = "/home/cbricout/scratch/"
    else:
        output_dir = "comet_downloads/"
    output_dir += f"{project_name}-{run_num}/"
    output_path = output_dir + f"{model_class.__name__}"
    pretrain_exp.download_model(
        model_class.__name__,
        output_path=output_path,
    )
    file_path = glob.glob(f"{output_path}/*.ckpt")[0]
    pretrained = PretrainingTask.load_from_checkpoint(checkpoint_path=file_path)

    if del_folers:
        shutil.rmtree(output_dir)
    return pretrained


def get_experiment_key(
    workspace_name: str, project_name: str, run_name: str, api_key: str = COMET_API_KEY
) -> str | None:
    """Find experiment key for a run, if it exists

    Args:
        api_key (str): Comet API key
        workspace_name (str): Comet workspace to use
        project_name (str): Comet project to use
        run_name (str): specific experiment to search

    Returns:
        str|None: experiment key, None if not existing
    """
    api = comet_ml.api.API(
        api_key=api_key,
    )
    exp = api.get_experiment(
        workspace=workspace_name, project_name=project_name, experiment=run_name
    )
    if exp:
        return exp.key
    return None


def log_figure_comet(figure: Figure, name: str, exp: comet_ml.Experiment):
    """Log figure to comet experiment

    Args:
        figure (Figure): Figure to log
        name (str): Figure name
        exp (comet_ml.Experiment): Comet experiment
    """
    with tempfile.NamedTemporaryFile() as img_file:
        figure.savefig(img_file)
        exp.log_image(
            img_file.name,
            name,
        )
