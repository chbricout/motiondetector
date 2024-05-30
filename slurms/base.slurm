#!/bin/bash
#SBATCH --job-name=pretrain-$model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --account=def-sbouix
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
    -n  --max_epochs 1000 --learning_rate 1e-5 --batch_size $batch_size --beta 1 \
    --model $model --dataset $dataset
# Deactivate the virtual environment
deactivate
