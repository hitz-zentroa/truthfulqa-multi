from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from peft import PeftModel
import json
from trl import setup_chat_format
import torch

def main():

    file = '/gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/lm-evaluation-harness/output/en/pretrained=ai-forever__mGPT_truthfulqa-multi_gen_en.jsonl'

    with open(file) as f:
        data = json.load(f)

    print(data[0]['filtered_resps'])

    model_name = '/gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/train_llama3_judge/Judge-Llama-3-8B-eng'
    base_model = "meta-llama/Meta-Llama-3-8B"
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model.resize_token_embeddings(len(tokenizer))
    # model.load_adapter(model_name)
    # model, tokenizer = setup_chat_format(model, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True, return_dict=True,
        torch_dtype=torch.float16, device_map="auto")
    #model, tokenizer = setup_chat_format(model, tokenizer)
    # Merge adapter with base model
    model = PeftModel.from_pretrained(model, model_name)
    model = model.merge_and_unload()



    for line in data:
        instruction = 'Q: '+ line['doc']['question'] + line['filtered_resps'][0] + '\nTrue: '
        #text = format_chat_template(instruction)
        print(instruction)

        inputs = tokenizer(instruction, return_tensors="pt")
        inputs = inputs.to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=2)
        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        out = out[len(instruction):]
        print(out)



if __name__ == '__main__':
    main()