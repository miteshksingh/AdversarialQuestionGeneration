# Stanford Attentive Reader

The QA reader model used inside the DrQA system is known as the Stanford Attentive Reader.

## Set up
* Install DrQA: https://github.com/facebookresearch/DrQA/tree/master/

## How to run inference on CPU?

python scripts/reader/predict.py ~/www/DrQA/data/datasets/SQuAD-v1.1-dev.json --model ~/www/DrQA/data/reader/single.mdl --no-cuda --tokenizer spacy

## How to train the reader on a dataset?

* python scripts/reader/preprocess.py data/converted_data data/converted_data --split generated_ques_3670 --tokenizer spacy
* python scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000 --data-dir data/datasets/ --train-file augmented-squad-sar-unilm-languagechecker-4930pqa-processed-spacy.txt --dev-file SQuAD-v1.1-dev-processed-spacy.txt --checkpoint True

