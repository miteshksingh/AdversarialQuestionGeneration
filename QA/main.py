import argparse
import json
import time
import torch

from allennlp.predictors.predictor import Predictor
from os.path import basename

"""
Parsing Runtime Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_json', required=True, help='SQuAD dataset file path')
args = parser.parse_args()

input_json = args.input_json
ans_json = "predictions-" + basename(input_json)

cuda_device = 2

print("Loading BiDAF Model")
predictor = Predictor.from_path("./bidaf-elmo-model-2018.11.30-charpad.tar.gz", cuda_device = cuda_device)

print("Reading " + input_json)
with open(input_json) as f:  data = json.load(f)

ans = {}
N = len(data["data"])

print("Evaluating AllenNLP")
for i, d in enumerate(data["data"]):
    
    t0 = time.time()
    for paragraph in d["paragraphs"]:
        context = paragraph["context"]

        for qas in paragraph["qas"]:

            id = qas["id"]
            question = qas["question"]

            try:
                prediction = predictor.predict(passage = context, question = question)

                """
                Prediction: dict_keys([
                    'passage_question_attention', 'span_start_logits', 'span_start_probs', 
                    'span_end_logits', 'span_end_probs', 'best_span', 
                    'best_span_str', 'question_tokens', 'passage_tokens'
                ])
                """

                ans[id] = prediction['best_span_str']
            except RuntimeError:
                ans[id] = ""
                pass

    t1 = time.time()

    print("{}/{}: {} Time Taken: {} secs".format(i+1, N, d["title"], round((t1-t0))))

with open(ans_json, 'w') as outfile: json.dump(ans, outfile, indent=4, sort_keys=True)
