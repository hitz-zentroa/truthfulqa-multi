
import json
from scipy.stats import chisquare
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar 


def main():

    langs = ['es', 'ca', 'eu', 'gl']
    models = ['Mistral-7B-Instruct-v0.3', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct', 'gemma-2-27b-it', 
              'gemma-2-9b-it',  'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']




    for lang in langs:
        print('\n'+lang)
        
        for model in models:
            print('\n'+model)
            data = {'human':[], 'mt':[]}
            with open('judge/judge_output/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                    human = json.load(f)
            

            with open('judge/judge_output/MT-claude/'+model+'__'+lang+'__multi_gemma9b_instruct_truth_judge__results.jsonl') as f:
                    mt = json.load(f)
    

            for entry in human:
                label = entry['label'] 
                if label in ['yes', 'no']: 
                    data['human'].append(entry['label'])

            for entry in mt:
                label = entry['label'] 
                if label in ['yes', 'no']: 
                    data['mt'].append(entry['label'])

            #print('iaa of', lang, 'and model', model, ':', cohen_kappa_score(data['human'], data['mt']))

            data_to_list = [[Counter(data['mt'])['yes'], Counter(data['mt'])['no']], [Counter(data['human'])['yes'], Counter(data['human'])['no']]]
            print(data_to_list)
            p = chisquare(data_to_list[0], data_to_list[1])

            print('CHI-SQUARE', p)

        #print('MCNEMAR', mcnemar(data_to_list, exact=False)) 


            

            
        
        

if __name__ == '__main__':
    main()