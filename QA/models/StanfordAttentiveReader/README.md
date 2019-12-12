# Stanford Attentive Reader

The QA reader model used inside the DrQA system is known as the Stanford Attentive Reader.

## Set up
* Install DrQA: https://github.com/facebookresearch/DrQA/tree/master/

## How to run inference on CPU?

python scripts/reader/predict.py ~/www/DrQA/data/datasets/SQuAD-v1.1-dev.json --model ~/www/DrQA/data/reader/single.mdl --no-cuda --tokenizer spacy

## How to train the reader on a dataset?

* python scripts/reader/preprocess.py data/converted_data data/converted_data --split generated_ques_3670 --tokenizer spacy
* python scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000 --data-dir data/datasets/ --train-file augmented-squad-sar-unilm-languagechecker-4930pqa-processed-spacy.txt --dev-file SQuAD-v1.1-dev-processed-spacy.txt --checkpoint True

## SAR Training Configuration on SQuAD v1.1 Training Dataset

Reader Configuration

{
    "batch_size": 32,
    "checkpoint": true,
    "concat_rnn_layers": true,
    "cuda": true,
    "data_dir": "data/datasets/",
    "data_workers": 5,
    "dev_file": "data/datasets/SQuAD-v1.1-dev-processed-spacy.txt",
    "dev_json": "data/datasets/SQuAD-v1.1-dev.json",
    "display_iter": 25,
    "doc_layers": 3,
    "dropout_emb": 0.4,
    "dropout_rnn": 0.4,
    "dropout_rnn_output": true,
    "embed_dir": "/var/services/homes/miksingh/www/DrQA/data/embeddings",
    "embedding_dim": 300,
    "embedding_file": "/var/services/homes/miksingh/www/DrQA/data/embeddings/glove.840B.300d.txt",
    "expand_dictionary": false,
    "fix_embeddings": false,
    "gpu": -1,
    "grad_clipping": 10,
    "hidden_size": 128,
    "learning_rate": 0.1,
    "log_file": "/tmp/drqa-models/20191212-0de7dfdb.txt",
    "max_len": 15,
    "model_dir": "/tmp/drqa-models/",
    "model_file": "/tmp/drqa-models/20191212-0de7dfdb.mdl",
    "model_name": "20191212-0de7dfdb",
    "model_type": "rnn",
    "momentum": 0,
    "no_cuda": false,
    "num_epochs": 40,
    "official_eval": true,
    "optimizer": "adamax",
    "parallel": false,
    "pretrained": "",
    "question_layers": 3,
    "question_merge": "self_attn",
    "random_seed": 1013,
    "restrict_vocab": true,
    "rnn_padding": false,
    "rnn_type": "lstm",
    "sort_by_len": true,
    "test_batch_size": 128,
     "train_file": "data/datasets/SQuAD-v1.1-train-processed-spacy.txt",
    "tune_partial": 1000,
    "uncased_doc": false,
    "uncased_question": false,
    "use_in_question": true,
    "use_lemma": true,
    "use_ner": true,
    "use_pos": true,
    "use_qemb": true,
    "use_tf": true,
    "valid_metric": "f1",
    "weight_decay": 0
} ]
