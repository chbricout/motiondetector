sbatch --export=ALL,batch_size=20,model="BASE",dataset="AMPSCZ" --job-name=BASE-ampscz-finetune slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="AMPSCZ" --job-name=RES-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="AMPSCZ" --job-name=SFCN-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="AMPSCZ" --job-name=Conv5_FC3-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="AMPSCZ" --job-name=SERes-ampscz-finetune  slurms/finetune.slurm


sbatch --export=ALL,batch_size=20,model="BASE",dataset="AMPSCZ",freeze_encoder=true --job-name=BASE-ampscz-finetune slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="AMPSCZ",freeze_encoder=true --job-name=RES-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="AMPSCZ",freeze_encoder=true --job-name=SFCN-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="AMPSCZ",freeze_encoder=true --job-name=Conv5_FC3-ampscz-finetune  slurms/finetune.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="AMPSCZ",freeze_encoder=true --job-name=SERes-ampscz-finetune  slurms/finetune.slurm