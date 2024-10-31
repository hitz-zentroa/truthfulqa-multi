#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source /gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/new_env/bin/activate

# MODELOS QUE NO USAMOS
#sh experiments/run/experiment_gen.sh ai-forever mGPT 
#sh experiments/run/experiment_gen.sh facebook xglm-7.5B # no funciona bien
#sh experiments/run/experiment_gen.sh bigscience bloomz-7b1

# MODELOS QUE USAMOS

# sh experiments/run/experiment_flash.sh mistralai Mistral-7B-Instruct-v0.3
# sh experiments/run/experiment_flash.sh meta-llama Meta-Llama-3-8B-Instruct
# sh experiments/run/experiment_eager.sh google gemma-2-9b-it

# sh experiments/run/experiment_flash_no_instruct.sh meta-llama Meta-Llama-3-8B
# sh experiments/run/experiment_eager_no_instruct.sh google gemma-2-9b
# sh experiments/run/experiment_flash_no_instruct.sh mistralai Mistral-7B-v0.3

sh experiments/run/experiment_flash_no_instruct.sh meta-llama Llama-3.1-8B
sh experiments/run/experiment_flash.sh meta-llama Llama-3.1-8B-Instruct