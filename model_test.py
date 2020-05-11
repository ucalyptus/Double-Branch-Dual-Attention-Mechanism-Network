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

class BiLinearSKNet(nn.Module):
  def __init__(self, band, classes,reduction):
    super(BiLinearSKNet, self).__init__()
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

net = BiLinearSKNet(200,16,4).to(device)
from torchsummary import summary
print(summary(net,(1,9,9,200),batch_size=16))
