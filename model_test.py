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

class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.input_channels = input_channels
        self.features_size = self._get_final_flattened_size()
        self.fc4 = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)
    
    def _get_final_flattened_size(self):
        patch_size=11
        with torch.no_grad():
            x = torch.zeros((1, 1,patch_size, patch_size, self.input_channels))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            _, t, c, w, h = x.size()
        return t * c * w * h


    def forward(self, x):
        x = x.permute(0,1,3,4,2)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        
        x = x.view(-1, self.features_size)
        
        x = self.fc4(x)
        
        return x

net = Baseline(20,16).to(device)
from torchsummary import summary
print(summary(net,(1,20,11,11),batch_size=16))
