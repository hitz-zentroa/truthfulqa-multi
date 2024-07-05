#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs_new/%j.out
#SBATCH --error=logs_new/%j.err

source env/bin/activate

sh individual_experiments.sh ai-forever mGPT
sh individual_experiments.sh facebook xglm-7.5B
sh individual_experiments.sh bigscience bloomz-7b1