
AUTHOR=$1
MODEL=$2

for lang in ca es eu gl en 
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,attn_implementation=eager \
        --tasks truthfulqa-multi_gen_$lang \
        --batch_size auto \
        --log_samples \
        --device cuda \
        --output_path results/gen/$lang \
        --num_fewshot 6

    done