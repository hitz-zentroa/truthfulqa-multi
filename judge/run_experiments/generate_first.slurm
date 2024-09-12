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

source env/bin/activate

#HF_HOME='/ixadata/users/bcalvo/cache'

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/Judge-Llama-3-8B-Instruct-eng \
#                                         --base_model meta-llama/Meta-Llama-3-8B-Instruct \
#                                         --langs en es ca eu gl \
#                                         --instruct \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model allenai/truthfulqa-truth-judge-llama2-7B \
#                                         --langs en es ca eu gl \
#                                         --label truth

#python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                        --judge_model judge/judge_models/llama-3_truth_judge_new \
#                                        --langs en es ca eu gl \
#                                        --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/llama-3-info-new \
#                                         --langs en es ca eu gl \
#                                         --label info

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model allenai/truthfulqa-info-judge-llama2-7B \
#                                         --langs en es ca eu gl \
#                                         --label info

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/llama3.1.2_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
                                        --judge_model judge/judge_models/llama3.1.2_info_judge \
                                        --langs en es ca eu gl \
                                        --label info

python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
                                        --judge_model judge/judge_models/llama3.1_multi_truth_judge \
                                        --langs en es ca eu gl \
                                        --label truth


