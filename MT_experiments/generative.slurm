#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err
#SBATCH --parsable --dependency=afterany:21450

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/new_env/bin/activate

# MODELOS QUE USAMOS

#sh MT_experiments/run/experiment_flash.sh mistralai Mistral-7B-Instruct-v0.3
#sh MT_experiments/run/experiment_flash.sh meta-llama Meta-Llama-3-8B-Instruct
sh MT_experiments/run/experiment_eager.sh google gemma-2-9b-it
sh MT_experiments/run/experiment_flash.sh meta-llama Llama-3.1-8B-Instruct

# sh MT_experiments/run/experiment_flash_no_instruct.sh meta-llama Meta-Llama-3-8B
# sh MT_experiments/run/experiment_eager_no_instruct.sh google gemma-2-9b
# sh MT_experiments/run/experiment_flash_no_instruct.sh mistralai Mistral-7B-v0.3
# sh MT_experiments/run/experiment_flash_no_instruct.sh meta-llama Llama-3.1-8B