#!/bin/bash

#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl309/jamliu/conda/etc/profile.d/conda.sh
conda activate prompt-styler

# Get the current time
# time=`date +%Y%m%d-%H%M%S`
date_ymd=`date +%Y%m%d`
time=`date +%H%M%S`
echo "${date_ymd}-${time}"

python -u train_classifier.py "$@"
echo $(date +%Y%m%d-%H%M%S)

# Organise the logs under date time
log_dir=./logs/train_classifier/${date_ymd}
mkdir -p ${log_dir}
mv ./logs/${SLURM_JOB_ID}.out ${log_dir}
