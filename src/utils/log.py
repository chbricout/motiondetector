"""Module defining utility function for logging"""

import logging
import os

from rich.logging import RichHandler

from src import config


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
