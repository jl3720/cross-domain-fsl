#!/bin/bash

#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl309/jamliu/conda/etc/profile.d/conda.sh
conda activate cd_fsl

# Get the current time
time=`date +%Y%m%d`
echo $(date +%H%M%S-%Y%m%d)

# Run the pipeline
# N.B. passes all args to scripts, only works because scripts accept same args
python -u cross_domain_fsl/baselines/finetune_foundation_fc.py "$@"
echo $(date +%H%M%S-%Y%m%d)

# Organise the logs under date time
new_dir="./logs/metatest_backbones/finetune_foundation_fc/${time}"
mkdir -p ${new_dir}
mv ./logs/${SLURM_JOB_ID}.out ${new_dir}