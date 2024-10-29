AUTHOR=$1
MODEL=$2

for lang in en ca es eu gl
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,attn_implementation=eager \
        --tasks truthfulqa-multi_mc2_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda \
        --output_path results/mc2/$lang \
        --num_fewshot 6

    done