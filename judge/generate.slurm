#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source ../truthfulqa-multi/env/bin/activate

python3 judge/llm_evaluate.py --models Meta-Llama-3-8B-Instruct --judge_model judge/judge_models/Judge-Llama-3-8B-Instruct-eng # Meta-Llama-3-70B-Instruct gemma-2-27b-it

