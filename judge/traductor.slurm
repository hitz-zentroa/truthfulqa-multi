#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --mem=18G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/trad_%j.out
#SBATCH --error=logs/trad_%j.err

source ../Easy-Translate/env/bin/activate

for lang in  cat_Latn eus_Latn glg_Latn spa_Latn
    do
    python3 ../Easy-Translate/translate.py \
    --sentences_path judge/data/info_answers_for_translation.txt \
    --output_path judge/data/translations/info_answers_$lang.txt \
    --source_lang eng_Latn \
    --target_lang $lang \
    --model_name facebook/nllb-200-3.3B
    
    done