AUTHOR=$1
MODEL=$2

for lang in eu gl en ca es 
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,parallelize=True,attn_implementation="flash_attention_2" \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size 8 \
        --log_samples \
        --device cuda \
        --output_path results/mc2/$lang \
        --num_fewshot 6

    done