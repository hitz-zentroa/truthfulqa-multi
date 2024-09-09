import json
import sys

def main():
    data = []
    dataset = sys.argv[1] # info or truth
    with open('judge/data/finetune_'+dataset+'.jsonl') as f:
        for line in f:
            data.append(json.loads(line))

    questions = []
    answers = []
    print(data[0])
    remove_characters = {'truth': -6, 'info': -9}
    for line in data:
        qu, an = line['prompt'].split('\nA: ', 1)
        questions.append(qu[3:].replace('\n', ' '))
        answers.append(an[:remove_characters[dataset]].replace('\n', ' ')) # -6 per truth

    print(len(questions), len(answers))

    # with open('judge/data/info_questions_for_translation.txt', 'w') as o:
    #     for line in questions:
    #         o.write(line)
    #         o.write('\n')

    with open('judge/data/'+dataset+'_answers_for_translation.txt', 'w') as o:
        for line in answers:
            o.write(line)
            o.write('\n')


if __name__ == '__main__':
    main()