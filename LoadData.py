# -*- coding:utf-8 -*-
# Created by steve @ 17-10-19 上午8:40
'''
                   _ooOoo_ 
                  o8888888o 
                  88" . "88 
                  (| -_- |) 
                  O\  =  /O 
               ____/`---'\____ 
             .'  \\|     |//  `. 
            /  \\|||  :  |||//  \ 
           /  _||||| -:- |||||-  \ 
           |   | \\\  -  /// |   | 
           | \_|  ''\---/''  |   | 
           \  .-\__  `-`  ___/-. / 
         ___`. .'  /--.--\  `. . __ 
      ."" '<  `.___\_<|>_/___.'  >'"". 
     | | :  `- \`.;`\ _ /`;.`/ - ` : | | 
     \  \ `-.   \_ __\ /__ _/   .-` /  / 
======`-.____`-.___\_____/___.-`____.-'====== 
                   `=---=' 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
         佛祖保佑       永无BUG 
'''

import scipy.io as sio
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import matplotlib.pyplot as plt


class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.bn_input = nn.BatchNorm1d(10, momentum=0.5)
        self.fc1 = nn.Linear(10, 20)
        self.bn2 = nn.BatchNorm1d(20, momentum=0.5)
        self.fc2 = nn.Linear(20, 30)
        self.bn3 = nn.BatchNorm1d(30, momentum=0.5)
        self.fc3 = nn.Linear(30, 10)
        self.bn4 = nn.BatchNorm1d(10, momentum=0.5)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        '''
        10->20->30->10->2 /drop_out batch_norm/
        :param x:
        :return:
        '''
        # x = self.bn_input(x)
        x = F.relu(self.fc1(x))
        # x = self.bn2(x)
        # x = F.dropout(x, training=True)
        x = F.relu(self.fc2(x))
        # x = self.bn3(x)
        # x = F.dropout(x, training=True)
        x = F.relu(self.fc3(x))
        # x = self.bn4(x)
        # x = F.dropout(x, training=True)
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    dir_name = '/home/steve/Data/Shad/'

    # x_file = sio.loadmat(dir_name+'Mill_500_25.mat')
    x_file = sio.loadmat(dir_name + 'Sond.mat')
    print(x_file)

    # x = x_file['X']
    # y = x_file['Y']
    x = x_file['X_temp']
    y = x_file['Y_temp']



    x = x.transpose()
    y = y.transpose()

    x = x[:,:10]

    print(x.shape, type(x))
    print(y.shape, type(y))

    x = x.astype(np.float)
    y = y.astype(np.float)

    print('x', x[10, :])
    print('y', y[10, :])

    x = preprocessing.scale(x)
    y = preprocessing.scale(y)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

    x_train = torch.from_numpy(x_train)
    x_valid = torch.from_numpy(x_valid)
    y_train = torch.from_numpy(y_train)
    y_valid = torch.from_numpy(y_valid)

    train_dataset = TensorDataset(data_tensor=x_train, target_tensor=y_train)
    test_dataset = TensorDataset(data_tensor=x_valid, target_tensor=y_valid)
    train_loader = DataLoader(train_dataset, batch_size=10, num_workers=4, pin_memory=True)
    test_dataset = DataLoader(test_dataset, batch_size=10, num_workers=4, pin_memory=True)

    # train_dataloader = DataLoader()

    fullNet = FullNet()
    fullNet.cuda()

    optimization = torch.optim.Adam(fullNet.parameters())
    loss_func = torch.nn.MSELoss()
    for epoch in range(4):

        print('Epoch:', epoch)
        for step,(bx, by )in enumerate(train_loader):
            bx, by = Variable(bx), Variable(by)
            print(bx)
            pred = fullNet(bx)
            loss = loss_func(pred, by)
            optimization.zero_grad()
            loss.backward()
            optimization.step()
