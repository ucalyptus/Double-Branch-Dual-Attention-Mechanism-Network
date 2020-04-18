import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime

import sys
sys.path.append('../global_module/')
import network
import train
from generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter,load_gamo
from Utils import record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

print('-----Importing Dataset-----')

label_url = '/content/drive/My Drive/Y_indianPines.mat'

global Dataset  # UP,IN,KSC
dataset = 'IN'
Dataset = dataset.upper()
import scipy.io as sio
gt = np.array(sio.loadmat(label_url)['Y']).transpose(axis=1)
CLASSES_NUM = 16
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = int(input("Enter num of iterations "))
PATCH_LENGTH = 2
# number of training samples per class
#lr, num_epochs, batch_size = 0.0001, 200, 32
lr, num_epochs, batch_size = 0.0005, 200, 16
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1

INPUT_DIMENSION = 25
ALL_SIZE ,TRAIN_SIZE = 39280,39280 
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

for index_iter in range(ITER):
    print(f"ITER : {index_iter+1}")
    net = network.SSRN_network(25, CLASSES_NUM)
    optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0001)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE , test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                   INPUT_DIMENSION, batch_size, gt,'/content/drive/My Drive/X_indianPines.mat',label_url)
    
    tic1 = time.clock()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.clock()

    pred_test_fdssc = []
    tic2 = time.clock()
    with torch.no_grad():
        for X, y in test_iter:
            print(X.shape)
            X = X.to(device)
            net.eval() 
            y_hat = net(X)
            # print(net(X))
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.clock()
    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])

    torch.save(net.state_dict(), "./net/" + str(round(overall_acc_fdssc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/' + 'GAMO' + '_' + Dataset + '_' +'25'+ '_'  + str(VALIDATION_SPLIT)  + '.txt')


generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)
print("location=\"",end="")
print("./records/" + 'GAMO' + '_' + Dataset + '_' +'25'+ '_'   + str(VALIDATION_SPLIT)  + '.txt',end="")
print("\"")
