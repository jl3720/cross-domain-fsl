#!/bin/bash

#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl309/jamliu/conda/etc/profile.d/conda.sh
conda activate cd_fsl

# Get the current time
# time=`date +%Y%m%d-%H%M%S`
date_ymd=`date +%Y%m%d`
time=`date +%H%M%S`
echo "${date_ymd}-${time}"

python -u cross_domain_fsl/baselines/prompt_generator.py "$@"
echo $(date +%Y%m%d-%H%M%S)

# Organise the logs under date time
log_dir=./logs/learn_styles/${date_ymd}
mkdir -p ${log_dir}
mv ./logs/${SLURM_JOB_ID}.out ${log_dir}
