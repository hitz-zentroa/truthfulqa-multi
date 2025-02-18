import datasets
from datasets import load_dataset
from evaluate import load
import json
from sklearn.metrics import cohen_kappa_score
import csv
from collections import Counter

def main():

    langs = ['es', 'ca', 'eu', 'gl']
    models = ['gemma-2-27b-it']#'Mistral-7B-Instruct-v0.3', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it', 
              #'Meta-Llama-3-70B', 'gemma-2-27b', 'Meta-Llama-3-8B', 'gemma-2-9b',  'Mistral-7B-v0.3', 'Llama-3.1-8B', 'Llama-3.1-70B', 
              #'gemma-2-9b-it',  'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']


    check_different_answers_across_languages = []
    with open('analysis/compare_to_MT_translation/check_instances.csv', 'w') as o:
        w = csv.writer(o)
        for lang in langs:
            print('\n'+lang)
            data = {'human':[], 'mt':[]}
            for model in models:
                with open('judge/judge_output/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                        human = json.load(f)
                

                with open('judge/judge_output/MT-claude/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                        mt = json.load(f)
        

                for entry in human:
                    label = entry['label'] 
                    if label in ['yes', 'no']: 
                        data['human'].append(entry['label'])

                for entry in mt:
                    label = entry['label'] 
                    if label in ['yes', 'no']: 
                        data['mt'].append(entry['label'])

                #print('iaa of', lang, 'and model', model, ':', cohen_kappa_score(data['human'], data['mt']))

                count_MT_right = 0
                count_human_right = 0
                for entry in human:
                    for entry2 in mt:
                        if entry['doc_id'] == entry2['doc_id'] and entry['label']=='yes' and entry2['label']=='no':
                            count_human_right += 1
                        elif entry['doc_id'] == entry2['doc_id'] and entry['label']=='no' and entry2['label']=='yes':
                            count_MT_right += 1

                        if entry['doc_id'] == entry2['doc_id'] and entry['label'] != entry2['label']:
                            w.writerow([lang, entry['doc_id'], entry['filtered_resps'][0], entry['label'], entry2['filtered_resps'][0], entry2['label'], entry['doc']['question'], '\n'.join(entry['doc']['correct_answers'])])
                            check_different_answers_across_languages.append((entry['doc_id']))

                


                # TODO: extract the overlap of instances, also load the differents to a csv just in case anyone wants to take a look
                print(str(count_human_right)+'/817', 'humana true y MT untrue', lang, model)
                print(str(count_MT_right)+'/817', 'humana untrue y humano true', lang, model)


    #print(Counter(check_different_answers_across_languages))
    print(len(check_different_answers_across_languages))
    count_counts = []
    keep_ids = []
    for key, count in Counter(check_different_answers_across_languages).items():
        count_counts.append(count)
        if count == 3:
            keep_ids.append(key)
    print(Counter(count_counts))

    with open('analysis/compare_to_MT_translation/check_instances_3wrong.csv', 'w') as o:
        w = csv.writer(o)

        for lang in langs:
            print('\n'+lang)
            data = {'human':[], 'mt':[]}
            for model in models:
                with open('judge/judge_output/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                        human = json.load(f)
                

                with open('judge/judge_output/MT-claude/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                        mt = json.load(f)
            for line in keep_ids:
                w.writerow([lang, line, human[line]['filtered_resps'][0], human[line]['label'], mt[line]['filtered_resps'][0], mt[line]['label'], human[line]['doc']['question'], mt[line]['doc']['question'], '\n'.join(human[line]['doc']['correct_answers'])])

        



            

            
        
        

if __name__ == '__main__':
    main()