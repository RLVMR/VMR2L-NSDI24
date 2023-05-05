#!/bin/bash
#SBATCH --job-name=gmm
#SBATCH --account=fc_nonsta
#SBATCH --partition=savio2_1080ti
# savio2_1080ti savio2_gpu # config in https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
# ############################# change the value here
#SBATCH --output=out_print_%A_task_%a.out
#SBATCH --error=out_progress_%A_task_%a.err
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yunkai_zhang@berkeley.edu

## Command(s) to run:
source activate torch

nvidia-smi
module load cuda/10.1

CUDA_VISIBLE_DEVICES=0 python3 train.py