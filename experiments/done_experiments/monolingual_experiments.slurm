#!/bin/bash
#SBATCH --job-name=eval_truthfulQA
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs_new/%j.out
#SBATCH --error=logs_new/%j.err

source env/bin/activate

AUTHOR=projecte-aina
MODEL=FLOR-6.3B-Instructed

# for lang in ca es
#     do
#     lm_eval --model hf \
#         --model_args pretrained=$AUTHOR/$MODEL \
#         --tasks truthfulqa-multi_mc2_$lang \
#         --batch_size auto \
#         --log_samples \
#         --device cuda:0 \
#         --output_path output/$lang/results_$MODEL.json 

#     done

AUTHOR=proxectonos
MODEL=FLOR-1.3B-GL
lang=gl

lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda:0 \
        --output_path output/$lang/results_$MODEL.json 

AUTHOR=HiTZ
MODEL=latxa-7b-v1.1
lang=eu

lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda:0 \
        --output_path output/$lang/results_$MODEL.json 