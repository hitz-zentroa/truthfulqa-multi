import json
import csv
from collections import Counter
import glob
import os
from sklearn.metrics import cohen_kappa_score
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
    models = ['Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it', 'Meta-Llama-3-70B']

    print('model_name', '\t', '\t'.join(langs))

    for model in models:
        out_raw = []
        for lang in langs:

            # LOAD MANUAL EVALUATIONS
            if model == 'Meta-Llama-3-70B':
                manual_location = 'judge/manual_review/evaluated/no_instruct/'
            else:
                manual_location = 'judge/manual_review/evaluated/'
            with open(manual_location+lang+'.csv') as f:
                r = csv.reader(f)
                next(r)
                manual_eval = []
                for line in r:
                    manual_eval.append(line)


            # LOAD THE MC2 RESULTS
            try:
                with open(latest_file_find(model, find_owner(model), lang, 'mc2')) as f:
                    mc2_results = []
                    for line in f:
                        mc2_results.append(json.loads(line))

            except IndexError:
                print('NOT FOUND: '+model+' '+lang)
                continue

            mc2_iaa_labels = []
            manual_labels = []
            for line in manual_eval:
                for entry in mc2_results:
                    if str(entry['doc']['id']-1)+'__'+model == line[0]:  # TODO: ids in manual eval are wrong, they need a +1, for now I correct it here
                        manual_labels.append(line[5])

                        if entry['acc'] > 0.5: 
                            mc2_iaa_labels.append('yes')
                        else:
                            mc2_iaa_labels.append('no')

            out_raw.append(str(round(cohen_kappa_score(mc2_iaa_labels, manual_labels), 2)))

        print(model, '\t', '\t'.join(out_raw))



if __name__ == '__main__':
    main()