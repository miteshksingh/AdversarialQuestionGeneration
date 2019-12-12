import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input1', type=str)
parser.add_argument('input2', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

data = []
question_cnt = 0
with open(args.input1) as f1, open(args.input2) as f2:
    for line1, line2 in zip(f1, f2):
        line1 = line1.rstrip("\n")
        line2 = line2.rstrip("\n")
        words = line1.split('[SEP]')
        
        question_cnt = question_cnt + 1
        qas = []
        qas.append({"question": line2, "answers": [{"answer_start": 0, "text": words[1]}], "id": str(question_cnt)}) 
        paragraphs = []
        paragraphs.append({'context': words[0], 'qas': qas})
        content = {"title": "dummy", "paragraphs": paragraphs}
        data.append(content)

squad = {"data": data, "version": "1.1"}
with open(args.output, 'w') as f3:
    f3.write(json.dumps(squad))



