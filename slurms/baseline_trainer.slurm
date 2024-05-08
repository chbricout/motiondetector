#!/bin/bash
#SBATCH --job-name=try-decoder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mail-user=bricout.charles@outlook.com
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

# Load Python module (assuming it's installed on your cluster)
module load python
module load cuda
module load httpproxy

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate bowl virtual environment
source ~/bowl/bin/activate

srun python src/training/base_trainer.py \
    -n  --max_epochs 1000 --learning_rate 1e-5 --batch_size 20 --beta 1 \
    --use_decoder --model BASE
# Deactivate the virtual environment
deactivate
