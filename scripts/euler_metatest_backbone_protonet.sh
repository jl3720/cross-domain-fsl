#!/bin/bash

#SBATCH  --output=logs/%j.out
#SBATCH  --gpus=1
#SBATCH  --gres=gpumem:11000m
#SBATCH  --mem-per-cpu=4000
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=4

source /cluster/home/jamliu/.bashrc  # load cvl settings and module envs
source /cluster/home/jamliu/virtualenvs/cd_fsl/bin/activate  # venv

# Get the current time
time=`date +%Y%m%d`
echo $(date +%H%M%S-%Y%m%d)

# Run the pipeline
# N.B. passes all args to scripts, only works because scripts accept same args
python -u cross_domain_fsl/baselines/metatest_foundation_protonet.py "$@"
echo $(date +%H%M%S-%Y%m%d)

# Organise the logs under date time
new_dir="./logs/metatest_backbones/foundation_protonet/${time}"
mkdir -p ${new_dir}
mv ./logs/${SLURM_JOB_ID}.out ${new_dir}