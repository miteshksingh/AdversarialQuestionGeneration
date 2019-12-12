from allennlp.predictors.predictor import Predictor

print('Load the predictor only once. Takes a long time')
#predictor = Predictor.from_path('fine-grained-ner-model-elmo-2018.12.21.tar.gz')
predictor = Predictor.from_path('ner-model-2018.12.18.tar.gz')
print ('Loading done')

entities = predictor.predict(sentence="Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.")


sen_len = len(entities['words'])
ner = []

#print(entities['words'])
for i in range(sen_len):
    print(entities['words'][i], ' ', entities['tags'][i])

print (ner)
