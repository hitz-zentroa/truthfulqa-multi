
AUTHOR=$1
MODEL=$2

for lang in en ca es eu gl
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL \
        --tasks truthfulqa-multi_gen_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda \
        --output_path results/gen/$lang


    done