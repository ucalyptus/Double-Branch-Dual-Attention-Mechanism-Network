
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
        self.data = self.data.unsqueeze(1)
        self.targets = torch.Tensor(self.targets)
        self.targets = torch.transpose(self.targets,0,1)
        self.targets = self.targets.squeeze(1)
        print(self.data.shape)
        print(self.targets.shape)
        

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
      
      return self.data[idx,:,:,:] , self.targets[idx]



data_train = HyperSpectralDataset('/content/drive/My Drive/X_indianPines.mat','/content/drive/My Drive/Y_indianPines.mat',16)
train_loader = DataLoader(data_train, batch_size=16, shuffle=True)


    
def train(net, train_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    
    train_acc_list = []
    
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            print(X.shape)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print('y_hat', y_hat)
            # print('y', y)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)
        
        

        # 绘图部分
        train_loss_list.append(train_l_sum) # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,  time.time() - time_epoch))

        PATH = "./net_DBA.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break

        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0: # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

    d2l.set_figsize()
    d2l.plt.figure(figsize=(8, 8.5))
    train_accuracy = d2l.plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train_accuracy')
    # train_acc_plot = np.array(train_acc_plot)
    # for x, y in zip(num_epochs, train_acc_plot):
    #    d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    test_accuracy = d2l.plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('test_accuracy')
    # test_acc_plot = np.array(test_acc_plot)
    # for x, y in zip(num_epochs, test_acc_plot):
    #   d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    loss_sum = d2l.plt.subplot(223)
    loss_sum.set_title('train_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='red')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train loss')
    # ls_plot = np.array(ls_plot)

    test_loss = d2l.plt.subplot(224)
    test_loss.set_title('valida_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('valida loss')
    # ls_plot = np.array(ls_plot)

    d2l.plt.show()
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
