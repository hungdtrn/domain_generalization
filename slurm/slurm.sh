#!/bin/bash
#SBATCH --job-name=domaingen
#SBATCH --gres=gpu:v100:4
#SBATCH --partition=gpu
#SBATCH --mincpus=8

source ~/.bashrc
conda activate dassl
echo $@
$@