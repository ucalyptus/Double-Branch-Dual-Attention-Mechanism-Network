import torch
from torch import nn

import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

import sys
sys.path.append('../global_module/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Path2D(nn.Module):
    def __init__(self,band,classes):
        super(Path2D, self).__init__()
        self.name = 'Path2D'
        self.conv2d_1 = nn.Sequential(nn.Conv2d(200, 128, 3), 
                        nn.ReLU())
        
        self.conv2d_2 = nn.Sequential(nn.Conv2d(128, 64, 3),
                        nn.ReLU())
                        
        self.conv2d_3 = nn.Sequential(nn.Conv2d( 64,64, 3),
                        nn.ReLU())
        
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        return x

class Path3D(nn.Module):
    def __init__(self,band,classes):
        super(Path3D, self).__init__()
        self.name = 'Path3D'
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 8, (3,3,11)), 
                        nn.ReLU())
        
        self.conv3d_2 = nn.Sequential(nn.Conv3d(8, 16, (3,3,7)),
                        nn.ReLU())
                        
        self.conv3d_3 = nn.Sequential(nn.Conv3d( 16,64, (3,3,5)),
                        nn.ReLU())
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,1,3,4,2)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return torch.mean(x,dim=4)

        
class BiLinearSKNet(nn.Module):
  def __init__(self, band, classes,reduction):
    super(BiLinearSKNet, self).__init__()
    self.name = 'BiLinearSKNet'
    self.Path3D = Path3D(band,classes)
    self.Path2D = Path2D(band,classes)
    channel=64
    self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
    self.conv2 = nn.Conv2d(channel, channel, 3, padding=2, dilation=2, bias=True)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.conv_se = nn.Sequential(
        nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
        nn.ReLU(inplace=True)
    )
    self.conv_ex = nn.Sequential(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True))
    self.softmax = nn.Softmax(dim=1)
    self.fc = nn.Linear(576,classes)    

  def forward(self,x):
    x = x.view(-1,200,9,9)
    P2 = self.Path2D(x)
    P3 = self.Path3D(x)
    conv1 = P2.unsqueeze(dim=1)
    conv2 = P3.unsqueeze(dim=1)
    features = torch.cat([conv1, conv2], dim=1)
    U = torch.sum(features, dim=1)
    S = self.pool(U)
    Z = self.conv_se(S)
    attention_vector = torch.cat([self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z).unsqueeze(dim=1)], dim=1)
    attention_vector = self.softmax(attention_vector)
    V = (features * attention_vector).sum(dim=1)
    batch,_,_,_ = V.size()
    V = V.view(batch,-1)
    return self.fc(V)
