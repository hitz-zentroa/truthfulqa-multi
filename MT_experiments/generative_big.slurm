#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/new_env/bin/activate

sh MT_experiments/run/experiment_big.sh meta-llama Meta-Llama-3-70B-Instruct
sh MT_experiments/run/experiment_big_eager.sh google gemma-2-27b-it

#sh MT_experiments/run/experiment_big_no_instruct.sh meta-llama Meta-Llama-3-70B
#sh MT_experiments/run/experiment_big_eager_no_instruct.sh google gemma-2-27b

#sh MT_experiments/run/experiment_big_no_instruct.sh meta-llama Llama-3.1-70B
sh MT_experiments/run/experiment_big.sh meta-llama Llama-3.1-70B-Instruct