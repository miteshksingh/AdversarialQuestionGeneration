#!/usr/bin/env python3
import argparse
import json

import spacy

nlp = spacy.load("en_core_web_sm")
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write paragraph-answer pairs
para_cnt = 0
question_cnt = 0
topic_cnt = 0
with open(args.output, 'w') as f:
    for article in dataset['data']:
        topic_cnt = topic_cnt + 1
        for paragraph in article['paragraphs']:
            para_cnt = para_cnt + 1
            texts = [paragraph['context']]
            for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
                for ent in doc.ents:
                    question_cnt = question_cnt + 1
                    f.write(texts[0] + ' [SEP] ' + ent.text)
                    f.write('\n')
        print('question_cnt ', question_cnt, ' done ')
        #if topic_cnt >= 9:
            #break
        print('para_cnt ', para_cnt, ' done ')
    print('para_cnt ', para_cnt, ' done ')
    print('question_cnt ', question_cnt, ' done ')
    print('topic_cnt', topic_cnt, ' done ')



