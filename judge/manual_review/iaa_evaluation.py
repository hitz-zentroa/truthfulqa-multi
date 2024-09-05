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

        #load iaa/lang
        with open('judge/manual_review/iaa/'+lang+'.csv') as f:
            r = csv.reader(f)
            next(r)
            iaa = []
            for line in r:
                iaa.append(line)

        # find same instances and calculate iaa
        truth_labels = []
        iaa_truth_labels = []
        info_labels = []
        iaa_info_labels = []
        for line in iaa:
            iaa_truth_labels.append(line[5])
            iaa_info_labels.append(line[6])
            for entry in data:
                if entry[0] == line [0]:
                    truth_labels.append(entry[5])
                    info_labels.append(entry[6])
                    break

        print('\n', lang)
        print(len(truth_labels), len(info_labels), len(iaa_info_labels), len(iaa_truth_labels))
        print(cohen_kappa_score(truth_labels, iaa_truth_labels))
        print(cohen_kappa_score(info_labels, iaa_info_labels))

    

if __name__ == '__main__':
    main()