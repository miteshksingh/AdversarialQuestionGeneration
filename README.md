# AdversarialQuestionGeneration
Generate questions on which a given QA system will fail

## Question Generation

### Setup
0. Do not install nvidia/apex module

1. Do not use -fp16 -amp flags in the python decode_seq2seq command options
https://github.com/microsoft/unilm/issues/23

2. Filename: code/unilm/src/run_finetuned_custom.sh
-python2 qg/eval_on_unilm_tokenized_ref.py --out_file qg/output/qg.test.output.txt
-python2 qg/eval.py --out_file qg/output/qg.test.output.txt

3. Filename: code/unilm/src/pytorch_pretrained_bert/__init__.py
-from .tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer
+from .tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer, WhitespaceTokenizer

python setup.py install --user

4. Filename: code/unilm/src/biunilm/seq2seq_decoder.py
-model_recover = torch.load(model_recover_path)
+model_recover = torch.load(model_recover_path, map_location=torch.device('cpu'))

### Running Steps:

1. conda activate unilm
2. cd code/unilm/src
3. ./run_finetuned_custom.sh

## Named Entity Recognition

### Set Up
1. Install Allennlp
conda install -c conda-forge allennlp

2. Download NER pre-trained model from : https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz

## Binary Classifier

### Set Up

1. Follow the steps given on Infereset github page (https://github.com/facebookresearch/InferSent), and move the file 'question_classifier.py' in that folder.
Note: Only download 'FastText' model, as it is better than Glove due to partial word matching
