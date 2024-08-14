"""
File to regroup important constant for the project
"""

import os
from dotenv import load_dotenv

load_dotenv()

## SLURM
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"

## TRAINING
IM_SHAPE = (1, 160, 192, 160)
PROJECT_NAME = "estimate-motion-mm"

## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (0, 4)
MOTION_BIN_STEP = (MOTION_BIN_RANGE[1] - MOTION_BIN_RANGE[0]) / MOTION_N_BINS

## SSIM PARAMETERS
SSIM_N_BINS = 40
SSIM_BIN_RANGE = (0, 0.5)
SSIM_BIN_STEP = (SSIM_BIN_RANGE[1] - SSIM_BIN_RANGE[0]) / SSIM_N_BINS

## COMET
COMET_API_KEY = os.environ.get("COMET_API_KEY")

## GENERATE DATASET
GENERATE_ROOT_DIR = os.environ.get("GENERATE_ROOT_DIR", "/home/cbricout/scratch/")

# PLOTS
PLOT_DIR = "plots"
CONFIDENCE_FILTER = 0.95  # Min confidence for swarm plot filter
