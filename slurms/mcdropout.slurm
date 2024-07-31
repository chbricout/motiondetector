#!/bin/bash
#SBATCH --job-name=mcdropout
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --account=def-sbouix
#SBATCH -o ./logs/output-%x.%j.out # STDOUT


# Load Python module (assuming it's installed on your cluster)
module load python
module load cuda
module load httpproxy

# Activate bowl virtual environment
source ~/bowl/bin/activate

srun python src/eval/mcdropout.py

deactivate