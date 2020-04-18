
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from scipy import io 
import torch.utils.data
import scipy
from scipy.stats import entropy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
import scipy.io as sio
def loadData():
    data = sio.loadmat('/content/drive/My Drive/X_indianPines.mat')['X']
    labels = sio.loadmat('/content/drive/My Drive/Y_indianPines.mat')['Y']
    
    return data, labels
X,y = loadData()
print(X.shape,y.shape)


class HyperSpectralDataset(Dataset):
    """HyperSpectral dataset."""

    def __init__(self,data_url,label_url,nb_classes):
        
        self.data = np.array(scipy.io.loadmat(data_url)['X'])
        self.targets = np.array(scipy.io.loadmat(label_url)['Y'])
        
        
        
        self.data = torch.Tensor(self.data)
        self.data = self.data.squeeze(4)
        #self.targets  = np.transpose(self.targets,0,1)
        #self.targets = np.squeeze(self.targets,axis=1)
        #self.targets = np.eye(nb_classes)[self.targets]
        #self.targets = torch.Tensor(self.targets)
        self.targets = torch.Tensor(self.targets)
        self.targets = torch.transpose(self.targets,0,1)
        self.targets = torch.squeeze(1)
        print(self.data.shape)
        print(self.targets.shape)
        

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
      
      return self.data[idx,:,:,:] , self.targets[idx]



data_train = HyperSpectralDataset('/content/drive/My Drive/X_indianPines.mat','/content/drive/My Drive/Y_indianPines.mat',16)
train_loader = DataLoader(data_train, batch_size=16, shuffle=True)
  
print(data_train.__getitem__(0)[0].shape)
print(data_train.__len__())

