import json
import csv
#from sklearn.metrics import cohen_kappa_score
from collections import Counter

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it']
    judge_models = [{'name':'multi-inst-llama3.1', 'files_name': 'multi_llama3.1_instruct_truth_judge', 'label':'truth'},
                    {'name':'multi-inst-gemma9b', 'files_name': 'multi_gemma9b_instruct_truth_judge', 'label':'truth'}]

    # for each language
    
    for judge in judge_models:
        avg_models = {}
        for m in models:
            avg_models[m] = []
        avg_lang = {'en':[], 'es':[], 'ca':[], 'eu': [], 'gl': []}
        print('\njudge')
        print('model_name', '\t', 'judge_name', '\t\t', '\t'.join(langs))
        errors = 0
        for model in models:
            out_raw = []
            for lang in langs:
                try:
                    # load the results from the llm_evaluate.py 
                    with open('judge/judge_output/'+model+'__'+lang+'__'+judge['files_name']+'__results.jsonl') as f:
                        results = json.load(f)
                except FileNotFoundError:
                    #print('judge/judge_output/'+model+'__'+lang+'__'+judge['files_name']+'__results.jsonl')
                    out_raw.append('-')
                    continue



                # find the instances that are also in the manual evaluation
                judge_labels = []
                for entry in results:
                    label = entry['label'].split('\nTrue:')[-1]
                    if label in ['yes', 'no']:
                        judge_labels.append(entry['label'])
                    elif 'yes' in label: # if the labels are not exacly yes but contain it, change the label, first results were save with "True:"" in front
                                #print(lang, 'label:', entry['label'])
                        judge_labels.append('yes')
                    elif 'no' in label:
                        judge_labels.append('no')
                            #print(lang, 'label:', entry['label'])
                    else:
                            #print(lang, 'label:', entry['label']) # TODO: this has some instances, review what to do with this
                        errors += 1
                        judge_labels.append('nsnc')

                out_raw.append(str(round(Counter(judge_labels)['yes']/len(judge_labels), 2)))
                avg_lang[lang].append(round(Counter(judge_labels)['yes']/len(judge_labels), 2))
            avg_models[model].append(round(Counter(judge_labels)['yes']/len(judge_labels), 2))


            print(model[:10], '\t', judge['name'], '\t', '\t'.join(out_raw))
        #print('Average for language:')
        #for lang in langs:
        #    print(lang, str(round(sum(avg_lang[lang])/len(avg_lang[lang]), 2)))
        print('\nAverage for model:')
        for model in models:
            print(model, str(round(sum(avg_models[model])/len(avg_models[model]), 2)))
        print('Errors in the labels:', str(errors))
    

if __name__ == '__main__':
    main()