import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import InferSent
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader

class QuestionDataset(Dataset):

  def __init__(self, json_file, balance_classes=True):
    with open(json_file) as f: data = json.load(f)
    
    if balance_classes:
      """
      Sampling equal +ve and -ve training examples.
      """
      positive_samples = [s for s in data if s["label"] == 1]
      negative_samples = [s for s in data if s["label"] == 0]
      
      random.shuffle(positive_samples)
      samples = positive_samples[:len(negative_samples)] + negative_samples
    else:
      samples = data

    """
    Extracting Question and Label from samples
    """
    self.N = len(samples)
    self.labels = []

    questions = []
    answers = []
    for entry in samples:
      questions.append(entry["question"])
      answers.append(entry["answers"][0]["text"])

      self.labels.append(entry["label"])

    """
    Getting Embedding for each question
    """
    embedding_model = Infersent()
    embeddings = embedding_model.get(questions + answers)

    self.question_embeddings = embeddings[:self.N]
    self.answer_embeddings = embeddings[self.N:]

  def __len__(self):
    return self.N

  def __getitem__(self, idx):
    X = torch.from_numpy(
      self.question_embeddings[idx] * self.answer_embeddings[idx]
    ).type(torch.FloatTensor)
    Y = torch.tensor(self.labels[idx])

    return X, Y

class Infersent:

  def __init__(self):

    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {
      'bsize': 64, 
      'word_emb_dim': 300, 
      'enc_lstm_dim': 2048,
      'pool_type': 'max', 
      'dpout_model': 0.0, 
      'version': V
    }

    self.infersent = InferSent(params_model)
    self.infersent.load_state_dict(torch.load(MODEL_PATH))
    self.infersent.set_w2v_path('fastText/crawl-300d-2M.vec')

  def get(self, sentences):
    self.infersent.build_vocab(sentences, tokenize=True)

    return self.infersent.encode(sentences, tokenize=True)


#our class must extend nn.Module
class Classifier(nn.Module):
    def __init__(self):
      super(Classifier,self).__init__()

      # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer

      self.d1 = nn.Dropout(0.5)

      # This applies Linear transformation to input data. 
      self.fc1 = nn.Linear(4096, 16)

      # This applies linear transformation to produce output data
      self.fc2 = nn.Linear(16, 2)
        
    # This must be implemented
    def forward(self, x):
      x = self.d1(x)

      # Output of the first layer
      x = self.fc1(x)

      # Activation function is Relu. Feel free to experiment with this
      x = torch.relu(x)
      
      # This produces output
      x = self.fc2(x)
      
      return x
        
    # This function predicts the class on unnormalized scores, (0 or 1)        
    def predict(self, x):

      # Apply softmax to output. 
      pred = F.softmax(x)
      
      # Pick the class with maximum weight
      return torch.argmax(pred, dim=1)

# Copied from: https://github.com/svishwa/crowdcount-mcnn/blob/master/src/network.py

def weights_normal_init(model, dev=0.01):
  if isinstance(model, list):
    for m in model:
        weights_normal_init(m, dev)
  else:
    for m in model.modules():
      if isinstance(m, nn.Conv2d):                
        m.weight.data.normal_(0.0, dev)
        if m.bias is not None:
          m.bias.data.fill_(0.0)
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, dev)

def train():
  BATCH_SIZE = 512
  LR = 0.001
  MAX_EPOCHS = 100
  RANDOM_SEED = 786
  MODEL_NAME_PREFIX = "classifier-bert"

  torch.manual_seed(RANDOM_SEED)
  torch.cuda.manual_seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  random.seed(RANDOM_SEED)

  t0 = time.time()
  train_c = QuestionDataset("../QA/labelled-bert_train_predictions.json")
  val_c = QuestionDataset("../QA/labelled-bert_dev_predictions.json", balance_classes=False)
  t1 = time.time()

  print("Data loaded. Train Size: {} Val Size: {} Time taken: {} secs".format(
    train_c.__len__(), val_c.__len__(), round(t1-t0)))

  train_dataloader = DataLoader(train_c, batch_size=BATCH_SIZE, shuffle=True)
  val_dataloader = DataLoader(val_c, batch_size=BATCH_SIZE, shuffle=False)

  model = Classifier()
  weights_normal_init(model, dev=0.001)

  criterion = nn.CrossEntropyLoss()
  
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  stats = []

  for epoch in range(MAX_EPOCHS):

    t0 = time.time()
    stat = {}

    """
    Training Model.
    """
    train_epoch_loss = 0
    all_y_pred = []
    all_y_true = []

    model.train()
    for _, batch in enumerate(train_dataloader):
      inputs, labels = batch
      M = labels.shape[0]

      # Clear the previous gradients
      optimizer.zero_grad()

      # Predict the output for Given input
      y_pred = model.forward(inputs)

      # Compute Cross entropy loss
      loss = criterion(y_pred, labels)
        
      # Compute gradients
      loss.backward()
    
      # Adjust weights
      optimizer.step()

      # Stats
      train_epoch_loss += (loss.item() * M)
      all_y_pred += model.predict(y_pred).tolist()
      all_y_true += labels.tolist()

    stat["loss"] = train_epoch_loss/train_c.__len__()
    stat["train_acc"] = accuracy_score(all_y_pred, all_y_true)

    """
    Predicting output on validation data.
    """
    all_y_pred = []
    all_y_true = []
    model.eval()
    for _, batch in enumerate(val_dataloader):
      inputs, labels = batch

      with torch.no_grad(): all_y_pred += model.predict(model.forward(inputs)).tolist()
      all_y_true += labels.tolist()

    stat["val_acc"] = accuracy_score(all_y_true, all_y_pred)
    stat["confusion_matrix"] = confusion_matrix(all_y_true, all_y_pred).tolist()
    stat["precision"] = precision_score(all_y_true, all_y_pred)
    stat["recall"] = recall_score(all_y_true, all_y_pred)
    stat["f1"] = f1_score(all_y_true, all_y_pred)

    stats.append(stat)

    model_name = "save/{}_{}.pth".format(MODEL_NAME_PREFIX, epoch+1)
    torch.save(model.state_dict(), model_name)

    t1 = time.time()

    print("Epoch: {} Avg. Epoch Loss: {} Train Acc: {} Val Acc: {} Time taken: {} secs".format(
        epoch+1, stat["loss"], stat["train_acc"], stat["val_acc"], round((t1-t0))))

  with open("stats-bert-SQuAD.json", 'w') as outfile: json.dump(stats, outfile, indent=4, sort_keys=True)

  # Plotting Epoch Loss
  plt.plot([e["loss"] for e in stats])
  plt.xlabel("Epochs")
  plt.ylabel("Epoch Loss")
  plt.show()

  # Plotting Validation Accuracy
  plt.plot([e["val_acc"] for e in stats])
  plt.xlabel("Epochs")
  plt.ylabel("Validation Accuracy")
  plt.show()

  # Plotting Training Accuracy
  plt.plot([e["train_acc"] for e in stats])
  plt.xlabel("Epochs")
  plt.ylabel("Training Accuracy")
  plt.show()

def test():

  model_name = "save/{}_{}.pth".format("classifier-sar", 83)
  model = Classifier()
  model.load_state_dict(torch.load(model_name))
  model.eval()

  test_c = QuestionDataset("../QA/labelled-predictions-unilm-languagechecker-4930pqa.json", balance_classes=False)
  test_dataloader = DataLoader(test_c, batch_size=1024, shuffle=False)

  predicted_label = []
  for _, batch in enumerate(test_dataloader):
    inputs, _ = batch
    with torch.no_grad(): predicted_label += model.predict(model.forward(inputs)).tolist()

  with open("classifier-sar-labelled-predictions-unilm-languagechecker-4930pqa.json", 'w') as outfile: 
    json.dump(predicted_label, outfile, indent=4)

def main():
  train()
  # test()

if __name__ == '__main__':
  main()
