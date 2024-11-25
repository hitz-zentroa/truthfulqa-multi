import json
import sys

langs = {'es':'Spanish', 'ca':'Catalan', 'eu':'Basque', 'gl':'Galician'}

partition = sys.argv[1]

for lang, file in langs.items():
	with open('analysis/compare_to_MT_translation/data/'+file+'_'+partition+'.json') as f:
		data = json.load(f)


	for line in data:
		line['mc1_targets'] = {}
		line['mc1_targets']['choices'] = [line['best_answer']]+line['incorrect_answers']
		line['mc1_targets']['labels'] = 1*[1]+len(line['incorrect_answers'])*[0]
		#print(line['mc1_targets']['labels'])
		assert len(line['mc1_targets']['choices']) == len(line['mc1_targets']['labels'])
		line['mc2_targets'] = {}
		line['mc2_targets']['choices'] = line['correct_answers']+line['incorrect_answers']
		line['mc2_targets']['labels'] = len(line['correct_answers'])*[1]+len(line['incorrect_answers'])*[0]
		#print(line['mc2_targets']['labels'])
		assert len(line['mc2_targets']['choices']) == len(line['mc2_targets']['labels'])

		line['lang'] = lang

	with open('data/MT/'+lang+'/'+partition+'.json', 'w') as o:
		json.dump(data, o, indent=4)

