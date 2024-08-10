'''
File to regroup important constant for the project
'''

import os
from dotenv import load_dotenv

load_dotenv()

## SLURM
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"

## TRAINING
IM_SHAPE = (1, 160, 192, 160)
PROJECT_NAME = "estimate-motion-bigger-ds"
N_BINS = 40
BIN_RANGE = (-0.1, 2.5)
BIN_STEP = (BIN_RANGE[1] - BIN_RANGE[0]) / N_BINS
COMET_API_KEY = os.environ.get("COMET_API_KEY")
