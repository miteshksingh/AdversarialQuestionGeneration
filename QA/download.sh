# Allen NLP BiDAF model
wget https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz

# SQuAD 1.1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

# SQuAD 2.0
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# SQuAD 2.0 Evaluation Script
curl "https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/" > evaluate-v2.0.py
