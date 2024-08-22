"""
File to regroup important constant for the project
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

## NARVAL ENV
IS_NARVAL = os.getenv("IS_NARVAL", "False").lower() in ("true", "1", "t")

## SLURM
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"



## TRAINING
IM_SHAPE = (1, 160, 192, 160)
PROJECT_NAME = "estimate-motion-pretrain"

## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (-0.8, 4.8)
MOTION_BIN_STEP = (MOTION_BIN_RANGE[1] - MOTION_BIN_RANGE[0]) / MOTION_N_BINS

## SSIM PARAMETERS
SSIM_N_BINS = 50
SSIM_BIN_RANGE = (-0.08, 0.48)
SSIM_BIN_STEP = (SSIM_BIN_RANGE[1] - SSIM_BIN_RANGE[0]) / SSIM_N_BINS

## COMET
COMET_API_KEY = os.getenv("COMET_API_KEY")

## GENERATE DATASET
GENERATE_ROOT_DIR = os.getenv("GENERATE_ROOT_DIR", "/home/cbricout/scratch/")

# PLOTS
PLOT_DIR = "plots"
CONFIDENCE_FILTER = 0.95  # Min confidence for swarm plot filter
