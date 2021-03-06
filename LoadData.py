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

from logger import Logger


class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.bn_input = nn.BatchNorm1d(10, momentum=0.5)
        self.fc1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(20, momentum=0.5)
        self.fc2 = nn.Linear(20, 30)
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(30, momentum=0.5)
        self.fc3 = nn.Linear(30, 10)
        self.relu3 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(10, momentum=0.5)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        '''
        10->20->30->10->2 /drop_out batch_norm/
        :param x:
        :return:
        '''
        # x = self.bn_input(x)
        x = (self.fc1(x))
        # x = self.bn2(x)
        # x = F.dropout(x, training=True)
        x = (self.fc2(x))
        # x = self.bn3(x)
        # x = F.dropout(x, training=True)
        x = (self.fc3(x))
        # x = self.bn4(x)
        # x = F.dropout(x, training=True)
        x = self.fc4(x)
        return x


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def own_loss_function(pre_y, y):
    return (((pre_y - y) / y)).abs().mean()


if __name__ == '__main__':

    ''' 
    initial logger
    '''
    import time

    logger = Logger('./logs' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '/')

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

    x = x[:, :10]

    print(x.shape, type(x))
    print(y.shape, type(y))

    x = x.astype(np.float)
    y = y.astype(np.float)

    print('x', x[10, :])
    print('y', y[10, :])

    # x = preprocessing.scale(x)
    # y = preprocessing.scale(y)

    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    x = (x-x_min)/(x_max-x_min)-0.5
    y = (y-y_min)/(y_max-y_min)-0.5


    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

    y_src_mean = y.mean(axis=0)
    y_src_std = y.std(axis=0)

    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

    print('x mean:', x.mean(axis=0))
    print('x std:', x.std(axis=0))
    print('y mean:', y.mean(axis=0))
    print('y std:', y.std(axis=0))

    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()

    train_dataset = TensorDataset(data_tensor=x_train, target_tensor=y_train)
    test_dataset = TensorDataset(data_tensor=x_valid, target_tensor=y_valid)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=4, pin_memory=True)
    # test_dataset = DataLoader(test_dataset, batch_size=100, num_workers=4)

    # train_dataloader = DataLoader()

    fullNet = nn.Sequential(
        # nn.BatchNorm1d(10),
        nn.Linear(10, 30),
        nn.PReLU(),
        nn.Dropout(),
        # nn.BatchNorm1d(20),
        nn.Linear(30, 30),
        nn.ELU(),
        nn.Dropout(),
        # nn.BatchNorm1d(20),
        nn.Linear(30, 20),
        # nn.Tanh(),
        nn.SELU(),
        # nn.PReLU(),
        nn.Dropout(),
        # nn.BatchNorm1d(20),
        nn.Linear(20, 20),
        nn.SELU(),
        # nn.Dropout(),
        # nn.BatchNorm1d(20),
        nn.Linear(20, 20),
        nn.SELU(),
        nn.Linear(20, 2),
        # nn.Tanh()
    )


    # from collections import OrderedDict
    # fullNet = nn.Sequential(
    #     OrderedDict([
    #         ('bn1',nn.BatchNorm1d(10)),
    #         ('fc1',nn.Linear(10,30)),
    #         ('af1',nn.Tanh()),
    #         ('dp1',nn.Dropout()),
    #         ('bn2',nn.BatchNorm1d(30)),
    #         ('fc2',nn.Linear(30,30)),
    #         ('af2',nn.Tanh()),
    #         ('dp2',nn.Dropout()),
    #         ('bn3',nn.BatchNorm1d(30)),
    #         ('fc3',nn.Linear(30,30)),
    #         ('af3',nn.Tanh()),
    #         ('fc4',nn.Linear(30,2))
    #
    #     ])
    #
    # )

    # nn.init.xavier_uniform(fullNet.fc1.weight)
    # nn.init.xavier_uniform(fullNet.fc2.weight)
    # nn.init.xavier_uniform(fullNet.fc3.weight)
    # nn.init.xavier_uniform(fullNet.fc4.weight)
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('linear'))
            # nn.init.xavier_normal(m.bias)


    fullNet.apply(weights_init)

    fullNet.cuda()
    fullNet.train()
    print(fullNet)
    print(fullNet.parameters())

    # x_test = torch.from_numpy(x_valid).float()
    # y_test = torch.from_numpy(y_valid).float()
    x_test = Variable(x_valid).cuda()
    y_test = Variable(y_valid).cuda()

    # optimization = torch.optim.SGD(fullNet.parameters(),momentum=0.005,lr=0.001)
    optimization = torch.optim.Adam(fullNet.parameters(), lr=0.005)
    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.SmoothL1Loss()
    # loss_func = own_loss_function()
    # nn.init.xavier_uniform(fullNet.parameters())
    running_loss = 0.0
    run_loss_count = 0
    for epoch in range(1000):

        print('Epoch:', epoch)
        for step, (bx, by) in enumerate(train_loader):
            bx, by = Variable(bx).cuda(), Variable(by).cuda()
            # print(bx)
            # print(by)
            pred = fullNet(bx)
            loss = loss_func(pred, by)
            optimization.zero_grad()
            loss.backward()
            optimization.step()

            running_loss += loss.data[0]
            run_loss_count += 1
            if step % 200 == 1:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / float(run_loss_count)))
                logger.scalar_summary('loss', running_loss / float(run_loss_count), step + epoch * 7600)
                running_loss = 0.0
                run_loss_count = 0
                pred_test = fullNet(x_test)
                score = loss_func(pred_test, y_test)
                logger.scalar_summary('score', score.data[0], step + epoch * 7600)

                np.savetxt('tmp_y.txt', pred_test.data.cpu().numpy())
                np.savetxt('y.txt', y_test.data.cpu().numpy())

                error_average = (((pred_test - y_test))).abs().mean()

                # print(error_average.float().cpu()[0].numpy())
                logger.scalar_summary('error_avg', error_average.data.cpu().numpy()[0], step + epoch * 7600)
                print('error avg:', error_average.data.cpu().numpy()[0])

                tmp_y = y_test.data.cpu().numpy()
                tmp_y_pred = pred_test.data.cpu().numpy()

                # tmp_y = (tmp_y * y_src_std) + y_src_mean
                # tmp_y_pred = (tmp_y_pred * y_src_std) + y_src_mean
                tmp_y = (tmp_y+0.5)*(y_max-y_min)+y_min
                tmp_y_pred = (tmp_y_pred+0.5)*(y_max-y_min)+y_min


                pred_error = np.mean(np.abs((tmp_y - tmp_y_pred) / tmp_y))
                print('predicted y std:', tmp_y_pred.std(axis=0))
                print('y_src std:', tmp_y.std(axis=0))
                np.savetxt('tmp_y_src.txt', tmp_y_pred)
                np.savetxt('y_src.txt', tmp_y)
                print('pred_error:', pred_error)
                logger.scalar_summary('pred_error', pred_error, step + epoch * 7600)
