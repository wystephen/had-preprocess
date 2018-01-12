# -*- coding:utf-8 -*-
# Created by steve @ 18-1-12 下午6:22
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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

if __name__ == '__main__':
    file_lines = open('/home/steve/Data/HardIU/UN1_Data.log').readlines()

    time_array = np.zeros([len(file_lines), 1])
    imu_time_list = list()
    imu_index_list = list()
    uwb_time_list = list()
    uwb_index_list = list()
    # for line in file_lines:
    for index in range(len(file_lines)):
        line = file_lines[index]
        time_array[index, 0] = line.split('[')[1].split(']')[0]
        if 'AX' in line:
            imu_time_list.append(time_array[index, 0])
            imu_index_list.append(index)
        else:
            uwb_time_list.append(time_array[index, 0])
            uwb_index_list.append(index)

    imu_time_array = np.asarray(imu_time_list)
    imu_index_array = np.asarray(imu_index_list)
    uwb_time_array = np.asarray(uwb_time_list)
    uwb_index_array = np.asarray(uwb_index_list)
    plt.figure()
    # plt.plot(time_array,'.',label='all')
    # plt.plot()

    plt.plot(imu_index_array, imu_time_array, '.', label='imu')
    plt.plot(uwb_index_array, uwb_time_array, '.', label='uwb')
    plt.legend()
    plt.grid()

    plt.figure()

    plt.plot(imu_index_array[1:], (imu_time_array[1:] - imu_time_array[:-1]), '.',label='imu')
    # plt.plot(uwb_index_array[1:], uwb_time_array[1:] - uwb_time_array[:-1], label='uwb')

    plt.grid()
    plt.legend()

    plt.plot()
    plt.show()
