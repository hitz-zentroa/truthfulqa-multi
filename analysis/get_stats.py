import json
from collections import Counter
import glob
import os
from sklearn.metrics import cohen_kappa_score
import csv
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from utils.utils import find_owner

def latest_file_find(model, type_model, language, results, file='samples', type='jsonl'):
    list_of_files = glob.glob(r'results/'+results+'/'+language+'/'+type_model+'__'+model+'/'+file+'_*.'+type)
    latest_files = sorted(list_of_files, key=os.path.getctime, reverse=True)
    return latest_files[0]

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-9b-it', 'gemma-2-27b-it', 'Mistral-7B-Instruct-v0.3', 
              'Meta-Llama-3-8B', 'Meta-Llama-3-70B',  'gemma-2-9b', 'gemma-2-27b',  'Mistral-7B-v0.3', 
              'Llama-3.1-8B', 'Llama-3.1-8B-Instruct']
    judge_models = [#{'name':'multi-inst-llama3.1', 'files_name': 'multi_llama3.1_instruct_truth_judge', 'label':'truth'},
                    {'name':'multi-inst-gemma9b', 'files_name': 'multi_gemma9b_instruct_truth_judge', 'label':'truth'}]
    

    # for each language
    for judge in judge_models:
        #print('\n'+judge['name']+'\n')
        # avg_models = {}
        # for m in models:
        #     avg_models[m] = []
        avg_lang = {'en':[], 'es':[], 'ca':[], 'eu': [], 'gl': []}
        #print('\njudge')
        #print('model_name', '\t', 'judge_name', '\t\t', '\t'.join(langs))
        errors = 0
        out_csv_category = []
        out_csv_type = []

        for model in models:
            #out_raw = []
            for lang in langs:
                all_types={}
                all_categories={}

                #print('\n', model, lang)

                # LOAD THE JUDGE RESULTS
                try:
                    # load the results from the llm_evaluate.py 
                    with open('judge/judge_output/'+model+'__'+lang+'__'+judge['files_name']+'__results.jsonl') as f:
                        judge_results = json.load(f)
                except FileNotFoundError:
                    print('NOT FOUND: judge/judge_output/'+model+'__'+lang+'__'+judge['files_name']+'__results.jsonl')
                    continue

                judge_labels = []
                types = []
                categories = []
                for entry in judge_results:
                    label = entry['label']#.split('\nTrue:')[-1]
                    types.append(entry['doc']['type'])
                    categories.append(entry['doc']['category'])
                    if label in ['yes', 'no']:
                        judge_labels.append(entry['label'])
                    elif 'yes' in label: # TODO: this should be removed when we have rerun all the judgements
                        judge_labels.append('yes')
                    elif 'no' in label:
                        judge_labels.append('no')
                    else:
                        judge_labels.append('other') # this should never happen
                        errors+=1
                #print(Counter(judge_labels)['yes']/817)

                # ANALYSE DIFFERENT RESULTS RELATIVE TO CATEGORY AND TYPE
                out = [['Model', model],['Language', lang]]
                
                for i,label in enumerate(judge_labels):

                    if types[i] not in all_types.keys():
                        all_types[types[i]] = [label]
                    else:
                        all_types[types[i]].append(label)

                    if categories[i] not in all_categories.keys():
                        all_categories[categories[i]] = [label]
                    else:
                        all_categories[categories[i]].append(label)


                all_types = {key: value for key, value in sorted(all_types.items())}
                all_categories = {key: value for key, value in sorted(all_categories.items())}

                for key, item in all_types.items():
                    out.append([key, float(round(Counter(item)['yes']/(len(item)), 2))])

                for key, item in all_categories.items():
                    out.append([key, float(round(Counter(item)['yes']/(len(item)), 2))])

                # with open('analysis/by_category/'+lang+'__'+model+'.csv', 'w') as o:
                #     w = csv.writer(o)
                #     w.writerows(out)

                if not out_csv_category:
                    out_csv_category.append(['model', 'language']+[key for key in all_categories.keys()])
                    out_csv_category.append([model, lang]+[float(round(Counter(item)['yes']/(len(item)), 2)) for item in all_categories.values()])
                else:
                    out_csv_category.append([model, lang]+[float(round(Counter(item)['yes']/(len(item)), 2)) for item in all_categories.values()])

                if not out_csv_type:
                    out_csv_type.append(['model', 'language']+[key for key in all_types.keys()])
                    out_csv_type.append([model, lang]+[float(round(Counter(item)['yes']/(len(item)), 2)) for item in all_types.values()])
                else:
                    out_csv_type.append([model, lang]+[float(round(Counter(item)['yes']/(len(item)), 2)) for item in all_types.values()])
        

        with open('analysis/by_category.csv', 'w') as o:
            w = csv.writer(o)
            w.writerows(out_csv_category)

        with open('analysis/by_type.csv', 'w') as o:
            w = csv.writer(o)
            w.writerows(out_csv_type)


                # print(all_types)
                # print(types_with_different_label)
                # print(all_categories)
                # print(categories_with_different_label)

                # for key,item in all_types.items():
                #     print(key, round(Counter(item)['yes']/(len(item)), 2))

                # for key,item in all_categories.items():
                #     print(key, round(Counter(item)['yes']/(len(item)),2))

if __name__ == '__main__':
    main()