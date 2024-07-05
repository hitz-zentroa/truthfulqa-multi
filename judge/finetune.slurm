#!/bin/bash
#SBATCH --job-name=finetune_judge
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source env/bin/activate

python finetune.py --new_model 'Judge-Instruct-3'

