# AdversarialQuestionGeneration
Generate questions on which a given QA system will fail


## Named Entity Recognition

### Set Up
1. Install Allennlp
conda install -c conda-forge allennlp

2. Download NER pre-trained model from : https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz

3. Follow the steps given on Infereset github page (https://github.com/facebookresearch/InferSent), and move the file 'question_classifier.py' in that folder.
Note: Only download 'FastText' model, as it is better than Glove due to partial word matching
