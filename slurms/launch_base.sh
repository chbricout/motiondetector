sbatch --export=ALL,batch_size=20,model="BASE",dataset="AMPSCZ" --job-name=BASE-ampscz-base slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="AMPSCZ" --job-name=RES-ampscz-base  slurms/base.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="AMPSCZ" --job-name=SFCN-ampscz-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="AMPSCZ" --job-name=Conv5_FC3-ampscz-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="AMPSCZ" --job-name=SERes-ampscz-base  slurms/base.slurm
sbatch --export=ALL,batch_size=12,model="ViT",dataset="AMPSCZ" --job-name=ViT-ampscz-base  slurms/base.slurm

sbatch --export=ALL,batch_size=20,model="BASE",dataset="MRART" --job-name=BASE-mrart-base slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="MRART" --job-name=RES-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="MRART" --job-name=SFCN-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="MRART" --job-name=Conv5_FC3-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="MRART" --job-name=SERes-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=12,model="ViT",dataset="MRART" --job-name=ViT-mrart-base  slurms/base.slurm
