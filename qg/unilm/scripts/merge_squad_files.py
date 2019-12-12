import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input1', type=str)
parser.add_argument('input2', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

data = []
question_cnt = 0

with open(args.input1) as f1:
    dataset1 = json.load(f1)

with open(args.input2) as f2:
    dataset2 = json.load(f2)


dataset = {"data":[], "version": 1.1}

filtered_q = set([])
articles1 = []
for article in dataset1['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            if question not in filtered_q:
                question_cnt = question_cnt + 1
                qa['id'] = str(question_cnt)
                articles1.append(article)
                filtered_q.add(question)

dataset['data'] = articles1

filtered_q = set([])
articles2 = []
for article in dataset2['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            if question not in filtered_q:
                question_cnt = question_cnt + 1
                qa['id'] = str(question_cnt)
                articles2.append(article)
                filtered_q.add(question)

for article in articles2:
    dataset['data'].append(article)

print(len(dataset['data']))
with open(args.output, 'w') as f3:
    f3.write(json.dumps(dataset))



