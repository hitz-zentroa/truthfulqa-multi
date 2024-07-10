#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/env/bin/activate

sh experiments/run/experiment_mc2_big.sh meta-llama Meta-Llama-3-70B-Instruct
sh experiments/run/experiment_mc2_big.sh google gemma-2-27b-it