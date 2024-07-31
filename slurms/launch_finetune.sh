batch_size=12
model="ViT"
account="rrg-ebrahimi"

for array_id in 1 2 3 4 5; do
    sbatch --export=ALL,batch_size=$batch_size,model=$model,dataset="MRART",run_num=$array_id --job-name=$model-ft-MRART --account=$account slurms/finetune.slurm
    sbatch --export=ALL,batch_size=$batch_size,model=$model,dataset="AMPSCZ",run_num=$array_id --job-name=$model-ft-AMPSCZ --account=$account slurms/finetune.slurm
done
