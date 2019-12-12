#!/usr/bin/env python3.

"""A script to convert the QA system output to the unilm input format:

QA System Output Format 
'{
    "data": [
                {
                    "passage":"abcd",
                    "question": "xyzdb",
                    "answers": []
                },
                ....
            ]
}'

unilm input format

' passage tokens 1 [SEP] answer tokens 1'
' answer tokens 2 [SEP} answer tokens 2'

"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for entry in dataset['data']:
        passage = entry['passage']
        for answer in entry['answers']:
            f.write(passage + ' [SEP] ' + answer)
            f.write('\n')
        
