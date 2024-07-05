#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs_new/%j.out
#SBATCH --error=logs_new/%j.err

source env/bin/activate

#sh individual_experiments.sh meta-llama Meta-Llama-3-8B
sh individual_experiments.sh mistralai Mistral-7B-v0.1