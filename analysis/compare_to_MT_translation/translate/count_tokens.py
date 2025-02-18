import anthropic
from datasets import load_dataset
import yaml
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

with open('scripts/utils/config.yaml', 'r') as file:
    yaml_config = yaml.safe_load(file)


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=yaml_config['anthropic']['token'],
)


en_data = load_dataset('HiTZ/truthfulqa-multi', 'en')


LANG = 'Spanish'

input_json = {}
for line in en_data['validation']:
    print(line)
    input_json['id'] = line['id']
    input_json['question'] = line['question']
    input_json['best_answer'] = line['best_answer']
    input_json['correct_answers'] = line['correct_answers']
    input_json['incorrect_answers'] = line['incorrect_answers']

    JSON_OBJECT = str(input_json)

    prompt = f"You are tasked with translating specific fields of a JSON object from English to {LANG}. Here is the JSON object you will be working with:\n\n<json_object>\n{JSON_OBJECT}\n</json_object>\n\nYour task is to translate the following fields into {LANG}:\n- question\n- best_answer\n- correct_answers\n- incorrect_answers\n\nImportant guidelines:\n1. Maintain the original structure of the JSON object.\n2. Only translate the content of the specified fields.\n3. Do not translate proper nouns.\n4. If a field contains an array, translate each element of the array.\n5. Preserve any formatting or special characters present in the original text.\n\nIf you encounter any content that should not be translated or you're unsure about, leave it in its original form.\n\nProvide the entire translated JSON object as your output. Do not include any comments or explanations outside of the JSON object."

    print(prompt)

    response = client.beta.messages.count_tokens(
        betas=["token-counting-2024-11-01"],
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "user",
            "content": prompt
        },
        {
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            }
        ],
    )

    print(response.json())
    exit()


