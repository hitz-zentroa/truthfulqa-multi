# TODO: pillar la llista del javier, fer dos datasets global VS local i veure si els resultats son diferents
import json
import csv
#from sklearn.metrics import cohen_kappa_score
from collections import Counter

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it', 
              #'Meta-Llama-3-70B', 'gemma-2-27b', 'Meta-Llama-3-8B',
              #'gemma-2-9b', 'Mistral-7B-v0.3', 'Llama-3.1-8B', 'Llama-3.1-70B', 
              'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']
    models.sort(reverse=True)
    judge_models = [{'name':'multi-inst-gemma9b', 'files_name': 'multi_gemma9b_instruct_truth_judge', 'label':'truth'}]

    selected = []
    with open('analysis/cultural_nuances/veritasqa.csv') as f:
        r = csv.reader(f)
        next(r)
        for line in r:
            selected.append(int(line[1].split('_')[-1]))

    # for each language
    with open('analysis/cultural_nuances/results.csv', 'w') as o:
        w = csv.writer(o)
        w.writerow(['model', 'subset', 'judge_name', 'en', 'es', 'ca', 'eu', 'gl'])
    
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



                    judge_labels_veritas = []
                    judge_labels_local = []
                    for entry in results:
                        if entry['doc']['id'] in selected:
                            if entry['label'] in ['yes', 'no']: 
                                judge_labels_veritas.append(entry['label'])
                            else:
                                errors += 1
                                judge_labels_veritas.append('nsnc')
                        else:
                            if entry['label'] in ['yes', 'no']: 
                                judge_labels_local.append(entry['label'])
                            else:
                                errors += 1
                                judge_labels_local.append('nsnc')
                    for subset in [judge_labels_local, judge_labels_veritas]:
                        out_raw.append(round((Counter(subset)['yes']/len(subset)*100), 2))
                        avg_lang[lang].append(round(Counter(subset)['yes']/len(subset), 2))
                #avg_models[model].append(round(Counter(judge_labels)['yes']/len(judge_labels), 2))
                print(len(out_raw))


                print(model, 'local', '\t', judge['name'], '\t', out_raw[0], out_raw[2], out_raw[4], out_raw[6], out_raw[7])
                print(model, 'veritas', '\t', judge['name'], '\t', out_raw[1], out_raw[3], out_raw[5], out_raw[7], out_raw[9])
                w.writerow([model, 'local', judge['name'], out_raw[0], out_raw[2], out_raw[4], out_raw[6], out_raw[7]])
                w.writerow([model, 'veritas', judge['name'], out_raw[1], out_raw[3], out_raw[5], out_raw[7], out_raw[9]])
                #w.writerow([model]+out_raw)
                #print('Average for language:')
                #for lang in langs:
                #    print(lang, str(round(sum(avg_lang[lang])/len(avg_lang[lang]), 2)))
                #print('\nAverage for model:')
                #for model in models:
                #    print(model, str(round(sum(avg_models[model])/len(avg_models[model]), 2)))
                #print('Errors in the labels:', str(errors))
    

if __name__ == '__main__':
    main()