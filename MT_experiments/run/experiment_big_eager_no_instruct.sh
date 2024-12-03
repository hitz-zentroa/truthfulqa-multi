
AUTHOR=$1
MODEL=$2

for lang in ca es eu gl
    do
    lm_eval --model hf \
        --model_args pretrained=$AUTHOR/$MODEL,parallelize=True,attn_implementation=eager \
        --tasks truthfulqa-multi-MT_gen_$lang \
        --batch_size 8 \
        --log_samples \
        --device cuda \
        --output_path results-MT/gen/$lang \
        --num_fewshot 6

    done