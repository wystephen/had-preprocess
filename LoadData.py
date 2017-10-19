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


if __name__ == '__main__':
    dir_name = '/home/steve/Data/Shad/'

    # x_file = sio.loadmat(dir_name+'Mill_500_25.mat')
    x_file = sio.loadmat(dir_name+ 'Sond.mat')
    print(x_file)

    # x = x_file['X']
    # y = x_file['Y']
    x = x_file['X_temp']
    y = x_file['Y_temp']

    x = x.transpose()
    y = y.transpose()


    print(x.shape)
    print(y.shape)


    x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2)


    print(x_train.shape,x_valid.shape,y_train.shape,y_valid.shape)

