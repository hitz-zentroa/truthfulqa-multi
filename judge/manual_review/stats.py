import json
import csv
from sklearn.metrics import cohen_kappa_score
from collections import Counter

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']

    # for each language
    for lang in langs:
        print('\n', lang)
        for path in ['judge/manual_review/evaluated/', 'judge/manual_review/evaluated/no_instruct/']:
        # load the results evaluated/lang
            with open(path+lang+'.csv') as f:
                r = csv.reader(f)
                next(r)
                data = []
                for line in r:
                    data.append(line)

            stats_by_model = {}
            for entry in data:
                if entry[5] not in ['yes', 'no'] or entry[6] not in ['yes', 'no']: # SEE WRONGLY ANNOTATED INSTANCES
                    print(lang, entry[0], entry[5], entry[6])

                model = entry[0].split('__')
                #print(list(stats_by_model.keys()))
                if model[1] not in list(stats_by_model.keys()):
                    stats_by_model[model[1]] = {'truth': [], 'info':[]}
                stats_by_model[model[1]]['truth'].append(entry[5])
                stats_by_model[model[1]]['info'].append(entry[6])
            
            
            for model in list(stats_by_model.keys()):
                print('truth', model,  Counter(stats_by_model[model]['truth']))
                print('info', model, Counter(stats_by_model[model]['info']))
    

if __name__ == '__main__':
    main()