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

class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            #nn.Conv3d(in_channels, out_channels,kernel_size=kernel_size, padding=padding, stride=stride),
            MinMaxCNNLayer(in_channels,out_channels,kernelsize=kernel_size,paddinglength=padding,stride=stride),
            nn.ReLU()
        )
        self.conv2 = MinMaxCNNLayer(out_channels,out_channels,kernelsize=kernel_size,paddinglength=padding,stride=stride)
        #nn.Conv3d(out_channels, out_channels,kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = MinMaxCNNLayer(in_channels,out_channels,kernelsize=1,stride=stride,paddinglength=0)
            #nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
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
    
class SSRN_MINMAX(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_MINMAX, self).__init__()
        self.name = 'SSRN_MINMAX'
        self.conv1 = MinMaxCNNLayer(1,24,(1,1,7),(1,1,2),(0,0,0))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24*2, eps=0.001, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24*2, 24*2, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24*2, 24*2, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24*2, 24*2, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24*2, 24*2, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)


        self.conv2 = MinMaxCNNLayer(24*2, 128, paddinglength=(0, 0, 0),
                               kernelsize=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128*2, eps=0.001, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True)
        )

        self.conv3 = MinMaxCNNLayer(1, 24, paddinglength=(0, 0, 0),kernelsize=(3, 3, 128), stride=(1, 1, 1))

        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24*2, eps=0.001, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(6192, classes)  # ,
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
        
        return self.full_connection(x4),x4
    
class SSRN_LSTM(nn.Module):
    def __init__(self,band,classes):
        super(SSRN_LSTM, self).__init__()
        self.name = 'SSRN_LSTM'
        self.ssrn = SSRN_MINMAX(band,classes)
        self.rnn = nn.LSTM(
            input_size=6192,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64, classes)

    def forward(self, x):
        batch_size, time_steps ,height, width , channels= x.size()
        
        c_in = x
        _, c_out = self.ssrn(c_in)
        
        r_in = c_out.view(batch_size, time_steps, -1)
        
        r_out, (_, _) = self.rnn(r_in)
        
        r_out2 = self.linear(r_out[:, -1, :])
        
        return F.log_softmax(r_out2, dim=1)
