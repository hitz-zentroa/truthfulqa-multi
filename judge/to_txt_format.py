import json

def main():
    data = []
    with open('judge/data/finetune_info.jsonl') as f:
        for line in f:
            data.append(json.loads(line))

    questions = []
    answers = []
    print(data[0])
    for line in data:
        qu, an = line['prompt'].split('\nA: ', 1)
        questions.append(qu[3:].replace('\n', ' '))
        answers.append(an[:-9].replace('\n', ' ')) # -6 per truth

    print(len(questions), len(answers))

    # with open('judge/data/info_questions_for_translation.txt', 'w') as o:
    #     for line in questions:
    #         o.write(line)
    #         o.write('\n')

    with open('judge/data/info_answers_for_translation.txt', 'w') as o:
        for line in answers:
            o.write(line)
            o.write('\n')


if __name__ == '__main__':
    main()