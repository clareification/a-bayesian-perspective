
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
from pylab import rcParams
import re 

from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix

import torch as th
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, datasets
from torchvision.datasets import MNIST, FashionMNIST

class MLP(nn.Module):
    def __init__(self, n_hidden, input_shape):
        super(MLP, self).__init__()

        #self.rate = rate

        self.fc1 = nn.Sequential(nn.Linear(in_features = 28*28, 
                             out_features =200), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features = 200, 
                             out_features = 200), nn.ReLU())
        self.fc4 = nn.Linear(in_features= 200, 
                            out_features = 10)
        
        self.final_activation = th.nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.flatten(1, -1)
      
        x = self.fc1(x)      
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)

        output = self.final_activation(x)
        return {"probs":output, "logits":x}

def train_mod(model, device, train_loader, loss, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss_step = loss(output["logits"], target)
        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()
    
    return model 

#Class extracts subnetworks
class MyModel(nn.Module):
    def __init__(self, network):
        super(MyModel, self).__init__()
        image_modules = list(network.children())[:2] #all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        image =image.flatten(1, -1)
        a = self.modelA(image)
        return a


def XE_loss(model, X, Y):
  logits = model.predict_log_proba(X)
  return np.mean(-1.0*logits[range(len(Y)),Y])

def get_lin_accuracy(indices, X_train, X_test, Y_train, Y_test): 
  scaler = StandardScaler()
  X_tr = X_train[:,indices].cpu().detach().numpy()
  scaler.fit(X_tr)
  X_tr = scaler.transform(X_tr)
  brier_score, acc, XE = [0.0,0.0,0.0]

  for rep in range(1):
    reg = LogisticRegression().fit(X_tr, Y_train.squeeze())
    X_te = X_test[:,indices].cpu().detach().numpy()
    X_te = scaler.transform(X_te)
    preds = reg.predict(X_te)
    acc +=reg.score(X_te, Y_test.squeeze())
    XE += XE_loss(reg, X_te, Y_test)
    brier_score += brier_score_loss(Y_test.squeeze(), preds) 
  

  return brier_score, acc, XE

if __name__ == '__main__':

  n_mod =10

  trs = transforms.Compose([transforms.ToTensor()])
  train = FashionMNIST("./data", train=True, download=True, transform=trs)
  test = FashionMNIST("./data", train=False, download=True, transform=trs)

  test_loader = th.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)
  train_loader = th.utils.data.DataLoader(train, batch_size=128, shuffle=True)

  device = th.device("cpu")

  #Train 10 models 
  for j in range(0, n_mod):
    th.manual_seed(j)

    network = MLP(n_hidden= 200, input_shape = [None, 28, 28]).to(device)
    optimizer = th.optim.SGD(network.parameters(),lr= 0.05)
    loss = nn.CrossEntropyLoss().to(device)

    #train network 
    for epch in range(0, 101):
    
      if epch ==0:
        PATH = "./checkpoints/fmnist_model_mod"+ str(j)+ "_epch" +str(epch)
        th.save(network.state_dict(), PATH)

      network = train_mod(network, device, train_loader, loss, optimizer) 
      if epch >0 and epch %10==0:  
        PATH = "./checkpoints/fmnist_model_mod"+ str(j)+ "_epch" +str(epch)
        th.save(network.state_dict(), PATH)
      
    print("training done")

  #Evaluate sub-network performance
  train_loader = th.utils.data.DataLoader(train, batch_size=len(train), shuffle=True)
  for batch_idx, (data, target) in enumerate(train_loader):
    break
  data_train, target = data.to(device), target.to(device)
  Y_train = target.cpu().detach().numpy()
  for batch_idx, (data, target) in enumerate(test_loader):
    break
  data_test, target = data.to(device), target.to(device)
  Y_test = target.cpu().detach().numpy()


  n_size= 10 #number of nodes in a subnetwork
  res_acc = np.zeros((n_mod,10,200//n_size))
  res_b = np.zeros((n_mod,10,200//n_size))
  res_XE = np.zeros((n_mod,10,200//n_size))

  for k in range(n_mod):
    PATH = "./checkpoints/fmnist_model_mod"+ str(k)+ "_epch" +str(100)
    network = MLP(n_hidden= 200, input_shape = [None, 28, 28]).to(device)
    network.load_state_dict(th.load(PATH))
    final_weights = np.abs(network.fc4.weight.data.cpu().numpy())
    model = MyModel(network)
    X_train = model(data_train)
    X_test = model(data_test)
    for i in range(10):
      indices = np.argsort(final_weights[i,:], axis =-1)
      for j in range((200//n_size)):
        y_tr = 1*(Y_train ==i)
        y_te = 1*(Y_test ==i)
        ind_ = indices[j*n_size:(j+1)*n_size]
        b,a,x = get_lin_accuracy(ind_, X_train, X_test, y_tr, y_te)
        res_acc[k,i,j]=a
        res_b[k,i,j]=b
        res_XE[k,i,j]=x
    print(res_XE[k])
    for z in range(90,-10,-10):
      PATH = "./checkpoints/fmnist_model_mod"+ str(k)+ "_epch" +str(z)
      network = MLP(n_hidden= 200, input_shape = [None, 28, 28]).to(device)
      network.load_state_dict(th.load(PATH))
      model = MyModel(network)
      X_train = model(data_train)
      X_test = model(data_test)

      for i in range(10):
        indices = np.argsort(final_weights[i,:], axis =-1)
        for j in range((200//n_size)):
          y_tr = 1*(Y_train ==i)
          y_te = 1*(Y_test ==i)
          ind_ = indices[j*n_size:(j+1)*n_size]
          b,a,x = get_lin_accuracy(ind_, X_train, X_test, y_tr, y_te)
          res_acc[k,i,j]+=a
          res_b[k,i,j]+=b
          res_XE[k,i,j]+=x  

    
    with h5py.File("final_results_submodels" + str(k), 'w') as f:
        tr = f.create_group('res')
        tr.create_dataset('res_acc', data=res_acc)
        tr.create_dataset('res_b', data=res_b)
        tr.create_dataset('res_XE', data=res_XE)
        tr.create_dataset('weights', data=final_weights)
