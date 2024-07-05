import json
import random
import csv

random.seed(2024)

def find_owner(model):
    if model == 'mGPT':
        return 'ai-forever'
    elif model in ['Llama-2-7b-hf', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B-Instruct']:
        return 'meta-llama'
    elif model in ['Mistral-7B-Instruct-v0.2', 'Mistral-7B-v0.1']:
        return 'mistralai'
    elif model in ['xglm-7.5B']:
        return 'facebook'
    elif model in ['bloomz-7b1']:
        return 'bigscience'
    else:
        raise Exception("Model type not defined.") 

def main():
    input_path = '../lm-evaluation-harness/output/'
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['bloomz-7b1', 'Mistral-7B-Instruct-v0.2', 'Meta-Llama-3-70B-Instruct']

    indices = random.sample(list(range(0,817)), 100)
    #print(indices)

    for lang in langs:
        with open('manual_review/sample_'+lang+'.csv', 'w') as o:
            w = csv.writer(o)
            w.writerow(['id', 'question', 'correct_answers', 'incorrect_answers', 'model_answer', 'label'])
            for model in models:
                path = input_path+lang+'/pretrained='+find_owner(model)+'__'+model+'_truthfulqa-multi_gen_'+lang+'.jsonl'

                with open(path) as f:
                    data = json.load(f)

                inspect_data = [line for i, line in enumerate(data) if i in indices]

                for row in inspect_data:
                    w.writerow([row['doc_id'], 
                                row['doc']['question'],
                                '\n'.join(row['doc']['correct_answers']), 
                                '\n'.join(row['doc']['incorrect_answers']),
                                row['filtered_resps'][0]])

            


if __name__ == '__main__':
    main()