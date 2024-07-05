#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

export PATH=/usr/local/cuda-12.1/bin:$PATH

source env/bin/activate


sh experiments/experiment_big.sh meta-llama Meta-Llama-3-70B-Instruct