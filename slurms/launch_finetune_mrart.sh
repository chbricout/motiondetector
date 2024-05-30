sbatch --export=ALL,batch_size=20,model="BASE",dataset="MRART" --job-name=finetune-mrart-BASE slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="MRART" --job-name=finetune-mrart-RES  slurms/finetune.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="MRART" --job-name=finetune-mrart-SFCN  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="MRART" --job-name=finetune-mrart-Conv5_FC3  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="MRART" --job-name=finetune-mrart-SERes  slurms/finetune.slurm

sbatch --export=ALL,batch_size=20,model="BASE",dataset="MRART",freeze_encoder=true --job-name=finetune-mrart-BASE slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="MRART",freeze_encoder=true --job-name=finetune-mrart-RES  slurms/finetune.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="MRART",freeze_encoder=true --job-name=finetune-mrart-SFCN  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="MRART",freeze_encoder=true --job-name=finetune-mrart-Conv5_FC3  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="MRART",freeze_encoder=true --job-name=finetune-mrart-SERes  slurms/finetune.slurm
