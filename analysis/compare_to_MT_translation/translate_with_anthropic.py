import anthropic
from datasets import load_dataset
import json
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import logging
logging.basicConfig()
logger = logging.getLogger()

import sys
import yaml
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

with open('utils/config.yaml', 'r') as file:
    yaml_config = yaml.safe_load(file)


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=yaml_config['anthropic']['token'],
)

en_data = load_dataset('HiTZ/truthfulqa-multi', 'en')

langs = ['Spanish', 'Catalan', 'Basque', 'Galician']

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(12))
def completion_with_backoff(**kwargs):
    return client.messages.create(**kwargs)


for LANG in langs:
    errors = 0
    to_save = []
    with open('analysis/compare_to_MT_translation/'+LANG+'.jsonl') as file:
        for line in file:
            to_save.append(json.loads(line))
        
    with open('analysis/compare_to_MT_translation/'+LANG+'.jsonl', 'a') as o:
        #to_save = []
        for line in en_data['validation']:
            if line['id'] <= to_save[-1]['id']:
                continue
            print(line['id'], flush=True)
            input_json = {}
            input_json['id'] = line['id']
            input_json['question'] = line['question']
            input_json['best_answer'] = line['best_answer']
            input_json['correct_answers'] = line['correct_answers']
            input_json['incorrect_answers'] = line['incorrect_answers']

            JSON_OBJECT = json.dumps(input_json)

            prompt = f"You are tasked with translating specific fields of a JSON object from English to {LANG}. Here is the JSON object you will be working with:\n\n<json_object>\n{JSON_OBJECT}\n</json_object>\n\nYour task is to translate the following fields into {LANG}:\n- question\n- best_answer\n- correct_answers\n- incorrect_answers\n\nImportant guidelines:\n1. Maintain the original structure of the JSON object.\n2. Only translate the content of the specified fields.\n3. Do not translate proper nouns.\n4. If a field contains an array, translate each element of the array.\n5. Preserve any formatting or special characters present in the original text.\n\nIf you encounter any content that should not be translated or you're unsure about, leave it in its original form.\n\nProvide the entire translated JSON object as your output. Do not include any comments or explanations outside of the JSON object."

            message = completion_with_backoff(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
                ]
            )

            out = "{"+message.content[0].text

            try:
                out_json = json.loads(out)
            except ValueError:
                logger.warning('Not able to parse '+str(line['id'])+':\n'+out)
                errors+=1
                if errors > 5:
                    logger.warning('Not JSON output for '+LANG)
                    break
                continue
            to_save.append(out_json)
            o.write(json.dumps(out_json))
            o.write('\n')
            o.flush()
            time.sleep(1)
            #break

    with open('analysis/compare_to_MT_translation/'+LANG+'.json', 'w') as p:
        json.dump(to_save, p, indent=4)
    
