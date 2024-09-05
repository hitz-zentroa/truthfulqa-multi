import json
import csv
from sklearn.metrics import cohen_kappa_score
from collections import Counter

def main():
    langs = ['en', 'es', 'ca', 'eu', 'gl']

    # for each language
    for lang in langs:
        # load the results evaluated/lang
        with open('judge/manual_review/evaluated/'+lang+'.csv') as f:
            r = csv.reader(f)
            next(r)
            data = []
            for line in r:
                data.append(line)
        stats_by_model = {}
        for entry in data:
            model = entry[0].split('__')
            #print(list(stats_by_model.keys()))
            if model[1] not in list(stats_by_model.keys()):
                stats_by_model[model[1]] = {'truth': [], 'info':[]}
            stats_by_model[model[1]]['truth'].append(entry[5])
            stats_by_model[model[1]]['info'].append(entry[6])
        
        print('\n', lang)
        for model in list(stats_by_model.keys()):
            print('truth', model,  Counter(stats_by_model[model]['truth']))
            print('info', model, Counter(stats_by_model[model]['info']))
    

if __name__ == '__main__':
    main()