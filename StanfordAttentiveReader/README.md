1. Install DrQA
https://github.com/facebookresearch/DrQA/tree/master/scripts/reader

2. How to run on CPU?

python scripts/reader/predict.py ~/www/DrQA/data/datasets/SQuAD-v1.1-dev.json --model ~/www/DrQA/data/reader/single.mdl --no-cuda --tokenizer spacy
