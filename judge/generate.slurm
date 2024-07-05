#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source env/bin/activate

python llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-8B --judge_model Judge-Llama-3-8B-Instruct-eng

