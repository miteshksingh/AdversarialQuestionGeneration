# Named Entity Recognition

## Model

We provide scripts to run both Allen NLP NER and spaCy NER model.
We found spaCy to be fast and more accurate.

## Allen NLP Set Up

* Install Allennlp: conda install -c conda-forge allennlp
* Download NER pre-trained model from : https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz

## spaCy setup

* conda create --name ner
* conda activate ner
* Follow the steps at: https://spacy.io/usage/linguistic-features#named-entities
* conda deactivate ner

## Commands

* python ner_spacy_squad.py input_file_path output_file_path

* Example: python ner_spacy_squad.py input/SQuAD-small-demo.json output/unilm-input-SQuAD-small-demo.txt

* The script generates output in the format which is acceptable as an input to  UniLM answer aware question generation model.

