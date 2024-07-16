#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc2_%j.out
#SBATCH --error=logs/mc2_%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/env/bin/activate

sh experiments/run/experiment_mc2.sh meta-llama Meta-Llama-3-8B-Instruct
sh experiments/run/experiment_mc2.sh mistralai Mistral-7B-Instruct-v0.2
sh experiments/run/experiment_mc2.sh google gemma-2-9b-it
