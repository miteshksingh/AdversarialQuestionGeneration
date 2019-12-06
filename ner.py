from allennlp.predictors.predictor import Predictor

print('Load the predictor only once. Takes a long time')
predictor = Predictor.from_path('ner-model-2018.12.18.tar.gz')
print ('Loading done')

entities = predictor.predict(sentence="Nikhil went there")


sen_len = len(entities['words'])
ner = []

for i in range(sen_len):
  if entities['tags'][i] != 'O':
    ner.append(entities['words'][i])

print (ner)