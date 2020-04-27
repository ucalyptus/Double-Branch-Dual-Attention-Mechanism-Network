import torch
from torch import nn
from attontion import PAM_Module, CAM_Module
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MinMaxCNNLayer(nn.Module):
  def __init__(self,infeatures,outfeatures,kernelsize,stride,paddinglength):
    super(MinMaxCNNLayer,self).__init__()
    self.infeatures = infeatures
    self.outfeatures = outfeatures
    self.kernelsize = kernelsize
    self.padding = paddinglength
    self.stride = stride

    self.cnn = nn.Conv3d(self.infeatures,self.outfeatures,self.kernelsize,self.stride,self.padding)
      
  def forward(self,x):
    conv1 = self.cnn(x)
    conv2 = (-1) * conv1
    conv3 = torch.cat((conv1,conv2),dim=1)
    return conv3
    
class MinMaxCNNNetwork(nn.Module):
  def __init__(self):
    super(MinMaxCNNNetwork,self).__init__()
    self.minmax1 = MinMaxCNNLayer(3,32,5,1,2)
    self.minmax2 = MinMaxCNNLayer(64,32,5,1,2)
    self.minmax3 = MinMaxCNNLayer(64,64,5,1,2)
    self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(6272,64)
    self.fc2 = nn.Linear(64,10)
  def forward(self,x):
    out = self.relu(self.minmax1(x))
    out = self.pool(out)
    out = self.relu(self.minmax2(out))
    out = self.pool(out)
    out = self.relu(self.minmax3(out))
    out = out.view(-1,6272)
    out = self.relu(self.fc1(out))
    out = self.fc2(out)
    return F.log_softmax(out,dim=1)


#minmaxcnn = MinMaxCNNNetwork().to(device)


class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class SSRN_network(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_network, self).__init__()
        self.name = 'SSRN'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(24, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        x1 = self.batch_norm1(self.conv1(X))
        # print('x1', x1.shape)

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        # print(x10.shape)
        return self.full_connection(x4)
