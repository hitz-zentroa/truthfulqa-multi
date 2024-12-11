#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err
#SBATCH --parsable --dependency=afterany:21442

source new_env/bin/activate

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/Judge-Llama-3-8B-Instruct-eng \
#                                         --base_model meta-llama/Meta-Llama-3-8B-Instruct \
#                                         --langs en es ca eu gl \
#                                         --instruct \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-70B \
#                                         --judge_model allenai/truthfulqa-truth-judge-llama2-7B \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                        --judge_model judge/judge_models/llama-3_truth_judge_new \
#                                        --langs en es ca eu gl \
#                                        --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-70B \
#                                         --judge_model judge/judge_models/llama-3-info-new \
#                                         --langs en es ca eu gl \
#                                         --label info

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/llama3.1.2_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-70B \
#                                         --judge_model judge/judge_models/llama3.1.2_info_judge \
#                                         --langs en es ca eu gl \
#                                         --label info

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/llama3.1_multi_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/gemma9b_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it  \
#                                         --judge_model judge/judge_models/2-llama3.1_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth \
#                                         --instruct


# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/gemma9b_multi_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it \
#                                         --judge_model judge/judge_models/llama-3-multi_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it  \
#                                         --judge_model judge/judge_models/gemma9b_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth \
#                                         --instruct


# MODELO JUDGE FINAL (remember "--instruct" refers to the judge)

# python judge/run_experiments/llm_evaluate.py --models gemma-2-27b gemma-2-9b Mistral-7B-v0.3 Meta-Llama-3-8B Meta-Llama-3-70B Llama-3.1-8B Llama-3.1-70B \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth \
#                                         --instruct

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it gemma-2-9b-it Mistral-7B-v0.3-Instruct Llama-3.1-8B-Instruct Llama-3.1-70B-Instruct  \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth \
#                                         --instruct

# EVALUCION INFORMATIVENESS

# python judge/run_experiments/llm_evaluate.py --models 'Meta-Llama-3-8B-Instruct'  'Meta-Llama-3-8B' 'gemma-2-27b-it' 'Meta-Llama-3-70B-Instruct' 'Meta-Llama-3-70B' \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_info_judge \
#                                         --langs en es ca eu gl \
#                                         --label info \
#                                         --input_path results/gen/ \
#                                         --output_path judge/judge_output/ \
#                                         --instruct

# python judge/run_experiments/llm_evaluate.py --models 'gemma-2-27b' 'gemma-2-9b' 'Llama-3.1-70B' 'Llama-3.1-8B' Mistral-7B-v0.3 \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_info_judge \
#                                         --langs en es ca eu gl \
#                                         --label info \
#                                         --input_path results/gen/ \
#                                         --output_path judge/judge_output/ \
#                                         --instruct

# EVALUACION DEL DATASET CON MT Claude

# python judge/run_experiments/llm_evaluate.py --models 'Meta-Llama-3-8B-Instruct'  'Meta-Llama-3-8B' 'gemma-2-9b' 'gemma-2-9b-it' 'Mistral-7B-Instruct-v0.3' 'Mistral-7B-v0.3'   'Llama-3.1-8B' 'Llama-3.1-8B-Instruct' \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
#                                         --langs es ca eu gl \
#                                         --label truth \
#                                         --input_path results-MT-claude/gen/ \
#                                         --output_path judge/judge_output/MT-claude/ \
#                                         --instruct



# python judge/run_experiments/llm_evaluate.py --models 'Meta-Llama-3-70B-Instruct'  'Meta-Llama-3-70B' 'gemma-2-27b' 'gemma-2-27b-it' 'Llama-3.1-70B' 'Llama-3.1-70B-Instruct' \
#                                         --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
#                                         --langs es ca eu gl \
#                                         --label truth \
#                                         --input_path results-MT-claude/gen/ \
#                                         --output_path judge/judge_output/MT-claude/ \
#                                         --instruct


# EVALUACION DEL DATASET CON MT de NLLB

python judge/run_experiments/llm_evaluate.py --models 'Meta-Llama-3-8B-Instruct'  'gemma-2-9b-it' 'Mistral-7B-Instruct-v0.3' 'Llama-3.1-8B-Instruct' \
                                        --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
                                        --langs es ca eu gl \
                                        --label truth \
                                        --input_path results-MT/gen/ \
                                        --output_path judge/judge_output/MT-NLLB/ \
                                        --instruct



python judge/run_experiments/llm_evaluate.py --models 'Meta-Llama-3-70B-Instruct' 'gemma-2-27b-it' 'Llama-3.1-70B-Instruct' \
                                        --judge_model judge/judge_models/multi_gemma9b_instruct_truth_judge \
                                        --langs es ca eu gl \
                                        --label truth \
                                        --input_path results-MT/gen/ \
                                        --output_path judge/judge_output/MT-NLLB/ \
                                        --instruct


# MODELO JUDGE ADICIONAL

# python judge/run_experiments/llm_evaluate.py --models gemma-2-27b gemma-2-9b Mistral-7B-v0.3 Meta-Llama-3-8B Meta-Llama-3-70B \
#                                         --judge_model judge/judge_models/multi_llama3.1_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth 

# python judge/run_experiments/llm_evaluate.py --models Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct gemma-2-27b-it gemma-2-9b-it Mistral-7B-Instruct-v0.3  \
#                                         --judge_model judge/judge_models/multi_llama3.1_instruct_truth_judge \
#                                         --langs en es ca eu gl \
#                                         --label truth \
#                                         --instruct


# MODELO INFO

# python judge/run_experiments/llm_evaluate.py --models gemma-2-27b gemma-2-9b Mistral-7B-v0.3 Meta-Llama-3-8B \
#                                         --judge_model allenai/truthfulqa-info-judge-llama2-7B \
#                                         --langs en es ca eu gl \
#                                         --label info