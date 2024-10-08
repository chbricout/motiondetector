"""Module defining utility function for logging"""

import logging
import os
from matplotlib.figure import Figure
import torch
from rich.logging import RichHandler
from PIL import Image
from torchvision.transforms import ToPILImage
from monai.transforms.intensity.array import ScaleIntensity

from src import config


def save_array_as_gif(imgs: list[Image.Image], file_path: str):
    """Save an array of Pillow Image as a gif

    Args:
        volume (list[Image.Image]): images to save
        file_path (str): path to save gif to
    """

    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(file_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def save_volume_as_gif(volume: torch.Tensor, file_path: str):
    """Save a volume as a gif

    Args:
        volume (torch.Tensor): volume to save
        file_path (str): path to save gif to
    """
    convert = ToPILImage()
    scale_01 = ScaleIntensity(0, 1)
    scaled = scale_01(volume)
    imgs = [convert(img) for img in scaled]

    save_array_as_gif(imgs, file_path)


def get_run_dir(project_name: str, run_name: str) -> str:
    """Define a directory for a specific run, used to store checkpoint
    Check existence and create needed folder

    Args:
        project_name (str): Name of the project (usually same as comet)
        run_name (str): Unique identifier for the specific run

    Returns:
        str: run directory full path
    """
    if config.IS_NARVAL:
        root_dir = f"/home/cbricout/scratch/{project_name}"
    else:
        root_dir = f"/home/at70870/local_scratch/{project_name}"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    run_dir = f"{root_dir}/{run_name}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def lightning_logger():
    """Setup python logging for use with lightning and rich print"""
    log: logging.Logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    log.setLevel(level="INFO")
    log.addHandler(RichHandler())


def rich_logger():
    """Setup python logging for rich print"""
    logging.basicConfig(
        level="INFO", handlers=[RichHandler()], format="%(message)s", datefmt="[%X]"
    )


def log_figure(figure: Figure, exp_dir: str, name: str, root_dir=config.PLOT_DIR):
    """Log Figure to a local directory

    Args:
        figure (Figure): Figure to log
        exp_dir (str): directory corresponding to experiment
        name (str): file name
        root_dir (_type_, optional): root directory for every figures. Defaults to config.PLOT_DIR.
    """
    img_path = os.path.join(root_dir, exp_dir, name)
    figure.savefig(img_path)
    logging.debug("logged figure to %s", img_path)
