#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc2_%j.out
#SBATCH --error=logs/mc2_%j.err

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/new_env/bin/activate

# sh experiments/run/experiment_mc2.sh mistralai Mistral-7B-Instruct-v0.3
# sh experiments/run/experiment_mc2.sh meta-llama Meta-Llama-3-8B-Instruct
# sh experiments/run/experiment_mc2.sh google gemma-2-9b-it

# sh experiments/run/experiment_mc2_no_instruct.sh mistralai Mistral-7B-v0.3
# sh experiments/run/experiment_mc2_no_instruct.sh meta-llama Meta-Llama-3-8B
# sh experiments/run/experiment_mc2_eager_no_instruct.sh google gemma-2-9b

sh experiments/run/experiment_mc2_no_instruct.sh meta-llama Llama-3.1-8B
sh experiments/run/experiment_mc2.sh meta-llama Llama-3.1-8B-Instruct