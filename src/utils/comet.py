"""Module defining comet utility functions"""

import glob
import comet_ml
from src.config import COMET_API_KEY
from src.network.utils import parse_model
from src.training.lightning_logic import PretrainingTask


def get_pretrain_task(
    model_name: str, run_num: int, project_name: str, api_key: str = COMET_API_KEY
) -> PretrainingTask:
    """Retrieve pretrain task from comet, used for Finetuning

    Args:
        model_name (str): name of the model to download
        run_num (int): specific run num to load
        project_name (str): name of the comet project

    Returns:
        PretrainingTask: loaded pretrained task
    """
    api = comet_ml.api.API(
        api_key=api_key,
    )
    pretrain_exp = api.get("mrart", project_name, f"pretraining-{model_name}-{run_num}")
    model_class = parse_model(model_name)

    output_path = (
        f"/home/cbricout/scratch/{project_name}-{run_num}/{model_class.__name__}"
    )
    pretrain_exp.download_model(
        model_class.__name__,
        output_path=output_path,
    )
    file_path = glob.glob(f"{output_path}/*.ckpt")[0]
    pretrained = PretrainingTask.load_from_checkpoint(checkpoint_path=file_path)
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
