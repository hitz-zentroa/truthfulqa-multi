import json
import csv
from sklearn.metrics import cohen_kappa_score

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']
    models = ['gemma-2-27b-it', 'Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-8B-Instruct']
    judge_model = 'Judge-Llama-3-8B-Instruct-eng'

    # for each language
    for lang in langs:
        for model in models:

            # load the results from the llm_evaluate.py 
            with open('judge/output/'+model+'__'+lang+'__'+judge_model+'__results.jsonl') as f:
                results = json.load(f)

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
                        judge_labels.append(entry['label'])
                        manual_labels.append(line[5])

            # compare the evaluations of both informativeness and truthfulness
            print(cohen_kappa_score(judge_labels, manual_labels))
    

if __name__ == '__main__':
    main()