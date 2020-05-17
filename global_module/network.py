import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

import sys
sys.path.append('../global_module/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        self.name = 'ChenEtAl'

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = self.pool1(x)
        
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        
        x = self.pool2(x)
        
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        
        x = self.fc(x)
        return x
