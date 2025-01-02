#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint="a40"
#SBATCH --time=23:00:00
#SBATCH --job-name=samba
#SBATCH --output=outputs/slurms/%j.out

module load boltzmachine
conda activate samba
python "$@"
