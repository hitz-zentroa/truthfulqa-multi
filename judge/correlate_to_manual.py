import json
import csv
from sklearn.metrics import cohen_kappa_score

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it']
    judge_models = [{'name':'hf-llama2-truth', 'files_name': 'truthfulqa-truth-judge-llama2-7B', 'label':'truth'},
                    {'name':'new-llama3-truth', 'files_name': 'llama3-1_7B_truth_judge_final', 'label':'truth'}, # now called llama-3-truth-new
                    {'name':'new-llama3-info', 'files_name': 'llama-3-info-new', 'label':'info'},
                    {'name':'hf-llama2-info', 'files_name': 'truthfulqa-info-judge-llama2-7B', 'label':'info'},
                    {'name':'llama3.1-truth', 'files_name': 'llama3.1.2_truth_judge', 'label':'truth'},
                    {'name':'llama3.1-info', 'files_name': 'llama3.1.2_info_judge', 'label':'info'},
                    {'name':'multi-ll3.1-truth', 'files_name': 'llama3.1_multi_truth_judge', 'label':'truth'}]

    # for each language
    
    for judge in judge_models:
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

                # load the manual evaluation
                with open('judge/manual_review/evaluated/'+lang+'.csv') as f:
                    r = csv.reader(f)
                    next(r)
                    manual_eval = []
                    for line in r:
                        manual_eval.append(line)

                # find the instances that are also in the manual evaluation
                judge_labels = []
                manual_labels = []
                for line in manual_eval:
                    for entry in results:
                        if str(entry['doc']['id']-1)+'__'+model == line[0]:  # TODO: ids in manual eval are wrong, they need a +1, for now I correct it here
                            #print(line[1], entry['doc']['question'])
                            #print(line[0], line[5], line[6], entry['label'])
                            if judge['label'] == 'truth':
                                manual_labels.append(line[5])
                            elif judge['label'] == 'info':
                                manual_labels.append(line[6])

                            if entry['label'] in ['yes', 'no']:
                                judge_labels.append(entry['label'])
                            elif 'yes' in entry['label']: # if the labels are not exacly yes but contain it, change the label, first results were save with "True:"" in front
                                #print(lang, 'label:', entry['label'])
                                judge_labels.append('yes')
                            elif 'no' in entry['label']:
                                judge_labels.append('no')
                                #print(lang, 'label:', entry['label'])
                            else:
                                #print(lang, 'label:', entry['label']) # TODO: this has some instances, review what to do with this
                                errors += 1
                                judge_labels.append('nsnc')
                    

                # compare the evaluations of both informativeness and truthfulness
                #print(lang, len(judge_labels), len(manual_labels))
                if 'yes' not in judge_labels:
                    out_raw.append('no')
                elif 'no' not in judge_labels:
                    out_raw.append('yes')
                else:
                    out_raw.append(str(round(cohen_kappa_score(judge_labels, manual_labels), 2)))
            print(model[:10], '\t', judge['name'], '\t', '\t'.join(out_raw))
        print('Errors in the labels:', str(errors))
    

if __name__ == '__main__':
    main()