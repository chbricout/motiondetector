"""
File to regroup important constant for the project
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

## LAB ENV
LAB_DIR = os.getenv("LAB_DIR", "")

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

## COMET
COMET_API_KEY = os.getenv("COMET_API_KEY")

## GENERATE DATASET
GENERATE_ROOT_DIR = os.getenv("GENERATE_ROOT_DIR", "/home/cbricout/scratch/")

# PLOTS
PLOT_DIR = "plots"
CONFIDENCE_FILTER = 0.95  # Min confidence for swarm plot filter

# RAYTUNE
RAYTUNE_DIR = "/lustre07/scratch/cbricout/ray_results" if IS_NARVAL else "~/ray_results"
