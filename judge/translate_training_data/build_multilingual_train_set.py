import json
from datasets import load_dataset
import random

def main():
    langs = ['es', 'ca', 'eu', 'gl']

    translation_dict = {}
    # for each language
    for lang in langs:

        # load the jsons in english and the language, and associate same questions in a dictionary
        data_en = load_dataset('HiTZ/truthfulqa-multi', 'en')
        new_lang = load_dataset('HiTZ/truthfulqa-multi', lang)

        #print(data_en)
        translation_dict[lang] = {}
        for line in data_en['validation']:
            for entry in new_lang['validation']:
                if line['id'] == entry['id']:
                    translation_dict[lang][line['question'].strip()] = entry['question'].strip()

    print(translation_dict.keys())
    
    for dataset in ['truth', 'info']:
        # load the english dataset
        data = []
        with open('judge/data/finetune_'+dataset+'.jsonl') as f:
            for line in f:
                data.append(json.loads(line))
        new_data = data.copy()

        for lang in langs:
            # load the data/translations/answerts_LANG
            from_lang_to_lang_codes = {'ca':'cat_Latn', 'eu': 'eus_Latn', 'gl': 'glg_Latn', 'es': 'spa_Latn'}
            with open('judge/data/translations/'+dataset+'_answers_'+from_lang_to_lang_codes[lang]+'.txt') as f:
                translated_answers = f.readlines()

            #print(translation_dict)
            print(len(data), len(translated_answers))

            # use the associated dictionary to translate the questions
            
            for i,line in enumerate(data):
                qu, an = line['prompt'].split('\nA: ', 1)
                question = qu[3:].strip()

                try:
                    new_question = translation_dict[lang][question]
                except KeyError:
                    #print('NOT IN TRANSLATION DICT: '+ question, i)
                    new_question = 'NOT FOUND' # TODO: discover what happens with the question "Are all plants plants?"
                    #count_errors += 1

                # in order, the txt translations are the answers
                try:
                    answer = translated_answers[i]
                except IndexError:
                    print(len(translated_answers), i) 
                    answer = 'INDEX ERROR'
                
                new_data.append({"prompt": 'Q: '+new_question+'\nA: '+answer.strip(), "completion": line['completion']})

            #print(new_data)
            #print(len(new_data), count_errors)
            random.shuffle(new_data)

            
            # build the final json just as in the finetune_truth.jsonl and save
            with open('judge/data/finetune_'+dataset+'_multi.jsonl', 'w') as o:
                json.dump(new_data, o, indent=5)
    

if __name__ == '__main__':
    main()