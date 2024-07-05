import json
import csv

def main():
    multilingual_models=['mGPT', 'xglm-7.5B', 'bloomz-7b1']
    english_models=['Meta-Llama-3-8B-Instruct', 'Mistral-7B-Instruct-v0.2']
    monolingual_models=['FLOR-6.3B-Instructed', 'latxa-7b-v1.1', 'FLOR-1.3B-GL']
    languages = ['en', 'es', 'ca', 'eu', 'gl']
    big_models= ['Meta-Llama-3-70B-Instruct', 'latxa-70b-v1.1']
    no_instruct = ['Meta-Llama-3-8B', 'Mistral-7B-v0.1']

    out_table = [['models']+languages]
    all_data = {}
    for model in multilingual_models+english_models+monolingual_models+big_models+no_instruct:
        results_model=[model]
        for language in languages:
            try:
                with open('output/'+language+'/results_'+model+'.json') as f:
                    all_data[language] = json.load(f)

                results_model.append(round(all_data[language]['results']['truthfulqa-multi_mc2_'+language]['acc,none']*100, 2))
            except FileNotFoundError:
                results_model.append('-')
            
        out_table.append(results_model)

    print(out_table)
    with open('output/results.csv', 'w') as o:
        w=csv.writer(o)
        w.writerows(out_table)

if __name__ == "__main__":
    main()