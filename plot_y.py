# -*- coding:utf-8 -*-
# Created by steve @ 17-10-19 下午9:54
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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pre_y = np.loadtxt('tmp_y_src.txt')
    y = np.loadtxt('y_src.txt')
    plt.figure()

    for i in range(y.shape[1]):
        # plt.plot((pre_y[:, i] - y[:, i]), '-*', label=str(i) + 'pre')
        plt.plot(pre_y[:,i],'.',label=str(i)+'pre')
        plt.plot(y[:,i],'-',label=str(i))
        print(i, ':', (np.abs(pre_y[:, i] - y[:, i])/np.abs(y[:,i]) ).mean())
    # plt.plot(y[:,i],'-+',label=str(i)+'y')
    plt.grid()
    plt.legend()

    plt.figure()
    for i in range(y.shape[1]):
        plt.plot(np.abs(pre_y[:,i]-y[:,i]),label=str(i)+'err')
    plt.grid()
    plt.legend()

    plt.show()
