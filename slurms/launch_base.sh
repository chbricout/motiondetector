sbatch --export=ALL,batch_size=20,model="BASE",dataset="MRART" --job-name=BASE-mrart-base slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="RES",dataset="MRART" --job-name=RES-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=14,model="SFCN",dataset="MRART" --job-name=SFCN-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="Conv5_FC3",dataset="MRART" --job-name=Conv5_FC3-mrart-base  slurms/base.slurm
sbatch --export=ALL,batch_size=20,model="SERes",dataset="MRART" --job-name=SERes-mrart-base  slurms/base.slurm
