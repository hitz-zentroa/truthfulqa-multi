import datasets
from datasets import load_dataset
from evaluate import load
import csv


def get_only_the_sentences(instance):
    #print(len(instance['correct_answers']),len(instance['incorrect_answers']))
    sentences = [instance['question']]+[instance['best_answer']]+instance['correct_answers']+instance['incorrect_answers']
    #print(len(sentences))
    return sentences

def average(values):
    return sum(values)/len(values)

def main():

    langs = ['es', 'ca', 'eu', 'gl']
    errors = []

    en_data = load_dataset('HiTZ/truthfulqa-multi', 'en')

    for lang in langs:
        
        mt_translated = load_dataset('HiTZ/truthfulqa-multi-MT', lang)#, download_mode="force_redownload")
        human_translated = load_dataset('HiTZ/truthfulqa-multi', lang)#, download_mode="force_redownload")

        metric = load('bleurt')

        for i, gold_references in enumerate(human_translated['validation']):
            predictions = get_only_the_sentences(mt_translated['validation'][i])
            
            references = get_only_the_sentences(gold_references)
            
            try:
                assert len(references) == len(predictions)
            except AssertionError: # this should not happen anymore
                references_visibles = '\n'.join(references)
                predictions_visibles = '\n'.join(predictions)
                visible_en = '\n'.join(get_only_the_sentences(en_data['validation'][i]))
                errors.append([lang, i, references_visibles, predictions_visibles, visible_en])
                continue
            metric.add_batch(predictions=predictions, references=references)
        final_score = metric.compute()
        print('\n RESULT \n', lang, average(final_score['scores']))
        
    with open('analysis/compare_to_MT_translation/errors.csv', 'w') as o:
        w=csv.writer(o)
        for line in errors:
            w.writerow(line)

if __name__ == '__main__':
    main()