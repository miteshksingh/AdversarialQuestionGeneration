import json
import nltk
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import InferSent
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

class QuestionDataset(Dataset):

  def __init__(self, json_file, balance_classes=True):
    with open(json_file) as f: data = json.load(f)
    
    if balance_classes:
      """
      Sampling equal +ve and -ve training examples
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
    for entry in samples:
      questions.append(entry["question"])
      self.labels.append(entry["label"])

    """
    Getting Embedding for each question
    """
    embedding_model = Infersent()
    self.question_embeddings = embedding_model.get(questions)

  def __len__(self):
    return self.N

  def __getitem__(self, idx):
    X = torch.from_numpy(self.question_embeddings[idx]).type(torch.FloatTensor)
    Y = torch.tensor(self.labels[idx])

    return X, Y

class Infersent:

  def __init__(self):

    self.V = 2
    self.MODEL_PATH = 'encoder/infersent%s.pkl' % self.V
    self.params_model = {
      'bsize': 64, 
      'word_emb_dim': 300, 
      'enc_lstm_dim': 2048,
      'pool_type': 'max', 
      'dpout_model': 0.0, 
      'version': self.V
    }

    self.infersent = InferSent(self.params_model)
    self.infersent.load_state_dict(torch.load(self.MODEL_PATH))
    self.W2V_PATH = 'fastText/crawl-300d-2M.vec'
    self.infersent.set_w2v_path(self.W2V_PATH)

  def get(self, sentences):
    self.infersent.build_vocab(sentences, tokenize=True)

    return self.infersent.encode(sentences, tokenize=True)


#our class must extend nn.Module
class Classifier(nn.Module):
    def __init__(self):
      super(Classifier,self).__init__()

      # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
      # This applies Linear transformation to input data. 
      self.fc1 = nn.Linear(4096, 512)

      # This applies linear transformation to produce output data
      self.fc2 = nn.Linear(512, 2)
        
    # This must be implemented
    def forward(self, x):
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


def main():
  BATCH_SIZE = 256
  LR = 0.01
  MAX_EPOCHS = 100
  RANDOM_SEED = 600

  torch.manual_seed(RANDOM_SEED)
  torch.cuda.manual_seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  random.seed(RANDOM_SEED)

  train_c = QuestionDataset("../QA/labelled-predictions-train-v1.1.json")
  val_c = QuestionDataset("../QA/labelled-predictions-dev-v1.1.json")
  print("Train Size: {} Val Size: {}".format(train_c.__len__(), val_c.__len__()))

  train_dataloader = DataLoader(train_c, batch_size=BATCH_SIZE, shuffle=True)
  val_dataloader = DataLoader(val_c, batch_size=BATCH_SIZE, shuffle=False)

  model = Classifier()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  for epoch in range(MAX_EPOCHS):

    t0 = time.time()

    """
    Training Model.
    """
    train_epoch_loss = 0
    all_y_pred = []
    all_y_true = []
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

    train_acc = accuracy_score(all_y_pred, all_y_true)

    """
    Predicting output on validation data.
    """
    all_y_pred = []
    all_y_true = []
    for _, batch in enumerate(val_dataloader):
      inputs, labels = batch

      with torch.no_grad(): all_y_pred += model.predict(model.forward(inputs)).tolist()
      all_y_true += labels.tolist()

    val_acc = accuracy_score(all_y_pred, all_y_true)

    t1 = time.time()

    print("Epoch: {} Avg. Epoch Loss: {} Train Acc: {} Val Acc: {} Time taken: {} secs".format(
        epoch+1, train_epoch_loss/train_c.__len__(), train_acc, val_acc, round((t1-t0))))

if __name__ == '__main__':
  main()
