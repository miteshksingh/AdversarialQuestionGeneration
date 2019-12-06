import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from models import InferSent
import numpy as np
from sklearn.metrics import accuracy_score

class Data:
  def __init__(self):
    self.X = None
    self.y = None

    with open('sampledata/train.txt' ,'r') as f:
      self.X = f.readlines()
      self.X = [line.strip() for line in self.X]

    with open('sampledata/labels.txt' ,'r') as f:
      self.y = f.readlines()
      self.y = np.asarray([line.strip() for line in self.y], dtype='float32')


class Infersent:
  def __init__(self, data):
    self.V = 2
    self.MODEL_PATH = 'encoder/infersent%s.pkl' % self.V
    self.params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': self.V}
    self.infersent = InferSent(self.params_model)
    self.infersent.load_state_dict(torch.load(self.MODEL_PATH))
    self.W2V_PATH = 'fastText/crawl-300d-2M.vec'
    self.infersent.set_w2v_path(self.W2V_PATH)

    self.infersent.build_vocab(data.X, tokenize=True)
    self.embeddings = self.infersent.encode(data.X, tokenize=True)

    self.X_train = torch.from_numpy(self.embeddings).type(torch.FloatTensor)
    self.y_train = torch.from_numpy(data.y).type(torch.LongTensor)


#our class must extend nn.Module
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(4096,8192)
        
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(8192,2)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = torch.relu(x)
        #This produces output
        x = self.fc2(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


def main():
  data = Data()
  encoder = Infersent(data)

  model = Classifier()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  epochs = 100
  losses = []
  for i in range(epochs):
    #Precit the output for Given input
    y_pred = model.forward(encoder.X_train)
    #Compute Cross entropy loss
    loss = criterion(y_pred,encoder.y_train)
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

  print(accuracy_score(model.predict(encoder.X_train),encoder.y_train))


if __name__ == '__main__':
  main()