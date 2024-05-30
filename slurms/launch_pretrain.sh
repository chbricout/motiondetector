sbatch --export=ALL,batch_size=20,model="BASE" --job-name=pretrain-BASE slurms/pretrain.slurm
sbatch --export=ALL,batch_size=20,model="RES" --job-name=pretrain-RES  slurms/pretrain.slurm
sbatch --export=ALL,batch_size=14,model="SFCN" --job-name=pretrain-SFCN  slurms/pretrain.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3" --job-name=pretrain-Conv5_FC3  slurms/pretrain.slurm
sbatch --export=ALL,batch_size=20,model="SERes" --job-name=pretrain-SERes  slurms/pretrain.slurm
