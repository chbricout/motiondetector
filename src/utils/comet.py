"""Module defining comet utility functions"""

import glob
import logging
import os
import shutil
import tempfile
from typing import Type

import comet_ml
import torch
from matplotlib.figure import Figure

from src import config
from src.config import COMET_API_KEY, IM_SHAPE, PROJECT_NAME
from src.network.utils import parse_model
from src.training.pretrain_logic import MotionPretrainingTask, PretrainingTask


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
