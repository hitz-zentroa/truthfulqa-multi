AUTHOR=$1
MODEL=$2

for lang in  en eu ca es
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,parallelize=True,attn_implementation="flash_attention_2" \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size 8 \
        --log_samples \
        --device cuda \
        --output_path results/mc2/$lang \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --num_fewshot 6

    done