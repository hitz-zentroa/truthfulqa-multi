#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc2_%j.out
#SBATCH --error=logs/mc2_%j.err
#SBATCH --parsable --dependency=afterany:21104

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/new_env/bin/activate

# sh experiments/run/experiment_mc2_big.sh meta-llama Meta-Llama-3-70B-Instruct
# sh experiments/run/experiment_mc2_big_eager.sh google gemma-2-27b-it

# sh experiments/run/experiment_mc2_big_no_instruct.sh meta-llama Meta-Llama-3-70B
# sh experiments/run/experiment_mc2_big_eager_no_instruct.sh google gemma-2-27b

# llençar nomes si els resultats del judge son força millors
#sh experiments/run/experiment_mc2_big.sh meta-llama Llama-3.1-70B-Instruct
sh experiments/run/experiment_mc2_big_no_instruct.sh meta-llama Llama-3.1-70B
