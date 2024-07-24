import json
import random
import csv
import os
import re
import glob

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
    elif model in ['gemma-2-9b-it', 'gemma-2-27b-it']:
        return 'google'
    else:
        raise Exception("Model type not defined.") 
    
def latest_file_find(model, type_model, language):
    list_of_files = glob.glob(r'results/gen/'+language+'/'+type_model+'__'+model+'/samples_*.jsonl')
    latest_files = sorted(list_of_files, key=os.path.getctime, reverse=True)
    return latest_files[0]

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['gemma-2-27b-it', 'Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-8B-Instruct']

    indices = random.sample(list(range(0,817)), 100)
    #print(indices)
    

    for lang in langs:
        iaa_data = []
        
        with open('judge/manual_review/sample_'+lang+'.csv', 'w') as o:
            w = csv.writer(o)
            w.writerow(['id', 'question', 'correct_answers', 'incorrect_answers', 'model_answer', 'truthfulness', 'informativeness'])
            for i, model in enumerate(models):
                next_iaa_instance = i*2
                path = latest_file_find(model, find_owner(model), lang)
                #path = input_path+lang+'/'+find_owner(model)+'__'+model+'/'
                #print(path)
                #file = find_file(path)
                #print(file)

                data = []
                with open(path) as f:
                    for line in f:
                        data.append(json.loads(line))

                inspect_data = [line for i, line in enumerate(data) if i in indices]

                for e,row in enumerate(inspect_data):
                    w.writerow([str(row['doc_id'])+'__'+model, 
                                row['doc']['question'],
                                '\n'.join(row['doc']['correct_answers']), 
                                '\n'.join(row['doc']['incorrect_answers']),
                                row['filtered_resps'][0]])

                    #print(next_iaa_instance)

                    if e==next_iaa_instance:
                        iaa_data.append([str(row['doc_id'])+'__'+model, 
                                row['doc']['question'],
                                '\n'.join(row['doc']['correct_answers']), 
                                '\n'.join(row['doc']['incorrect_answers']),
                                row['filtered_resps'][0]])
                        next_iaa_instance+=6


        print(len(iaa_data))

        with open('judge/manual_review/iaa_'+lang+'.csv', 'w') as o:
            w = csv.writer(o)
            w.writerow(['id', 'question', 'correct_answers', 'incorrect_answers', 'model_answer', 'truthfulness', 'informativeness'])
            for instance in iaa_data:
                w.writerow(instance)

                

            


if __name__ == '__main__':
    main()