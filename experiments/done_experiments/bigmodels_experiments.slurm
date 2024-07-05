#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs_new/%j.out
#SBATCH --error=logs_new/%j.err

source env/bin/activate

# AUTHOR=meta-llama
# MODEL=Meta-Llama-3-70B-Instruct

# for lang in gl ca es en eu
#     do
#     lm_eval --model hf \
#         --model_args pretrained=$AUTHOR/$MODEL,parallelize=True \
#         --tasks truthfulqa-multi_mc2_$lang \
#         --batch_size auto \
#         --log_samples \
#         --device cuda \
#         --output_path output/$lang/results_$MODEL.json 

#     done

AUTHOR=HiTZ
MODEL=latxa-70b-v1.1
lang=eu

lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,parallelize=True \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda \
        --output_path output/$lang/results_$MODEL.json 


# AUTHOR=mistralai
# MODEL=Mixtral-8x7B-Instruct-v0.1

# for lang in gl ca es en eu
#     do
#     lm_eval --model hf \
#         --model_args pretrained=$AUTHOR/$MODEL \
#         --tasks truthfulqa-multi_mc2_$lang \
#         --batch_size auto \
#         --log_samples \
#         --device cuda:0 \
#         --output_path output/$lang/results_$MODEL.json \
#         --model_args parallelize=True

#     done