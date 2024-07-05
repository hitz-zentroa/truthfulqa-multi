import json
from transformers import MarianMTModel, MarianTokenizer
import ctranslate2
import pyonmttok
from huggingface_hub import snapshot_download
import torch
import gc
from tqdm import tqdm

def clean_input(line):
    remove_Q = line[2:]
    remove_True = remove_Q[:-6]
    remove_A = remove_True.split('\nA: ')
    return remove_A

def main():

    with open('/gaueko0/users/bcalvo/GSCRATCH/truthfulqa-multi/train_llama3_judge/data/finetune_truth.jsonl') as f:
        data=[]
        for line in f:
            data.append(json.loads(line))

    

    with open('data/multilingual_fintune_truth.jsonl', 'w') as f:
        for line in data:
            line['lang'] = 'en'
            f.write(json.dumps(line))
            f.write('\n')

        langs = ['ca', 'es', 'eu', 'gl']

        multilingual_data = data

        for lang in langs:

            #model_dir = snapshot_download(repo_id="projecte-aina/aina-translator-en-"+lang, revision="main")
            #tokenizer=pyonmttok.Tokenizer(mode="none", sp_model_path = model_dir + "/spm.model")
            #translator = ctranslate2.Translator(model_dir)
            
            model_name = "Helsinki-NLP/opus-mt-en-"+lang
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            for line in tqdm(data):
                new_line = {'prompt':'', 'completion':line['completion'], 'lang': lang}

                # TODO: remove the Q: and the A: before translating
                try:
                    prompt = clean_input(line['prompt'])
                except IndexError:
                    print(line['prompt'])
                    continue
                
                #tokenized=tokenizer.tokenize(prompt)
                #translated = translator.translate_batch([tokenized[0]])
                #print(tokenizer.detokenize(translated[0][0]['tokens']))
                #print(prompt)

                #generated = [model.generate(**tokenizer(s, padding=True)) for s in prompt]
                translated = model.generate(**tokenizer(prompt, return_tensors="pt", padding=True))
                sentences = [tokenizer.decode(s, skip_special_tokens=True) for s in translated]
                out = 'Q: ' + '\n A: '.join(sentences) + '\nTrue:'

                #new_line['prompt'] = tokenizer.detokenize(translated[0][0]['tokens'])
                new_line['prompt'] = out
                #print(new_line)

                multilingual_data.append(new_line)

        
                f.write(json.dumps(new_line))
                f.write('\n')

            print('Finished with', lang)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            

if __name__ == '__main__':
    main()

