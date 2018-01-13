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

import array
import re

if __name__ == '__main__':
    file_lines = open('/home/steve/Data/HardIU/UN1_Data.log').readlines()

    time_array = np.zeros([len(file_lines), 1])
    imu_time_list = list()
    imu_index_list = list()
    uwb_time_list = list()
    uwb_index_list = list()

    imu_buff = array.array('d')
    num_re_mag = re.compile("[-]{0,1}[0-9]{1,5}")

    # for line in file_lines:
    for index in range(len(file_lines)):
        line = file_lines[index]
        time_array[index, 0] = line.split('[')[1].split(']')[0]
        if 'AX' in line:
            imu_time_list.append(time_array[index, 0])
            imu_index_list.append(index)
            all_num = num_re_mag.findall(line, line.index(']'))
            # print("all num:",all_num)
            for n in all_num:
                imu_buff.append(float(n) / 32768.0)
        else:
            uwb_time_list.append(time_array[index, 0])
            uwb_index_list.append(index)

    imu_time_array = np.asarray(imu_time_list)
    imu_index_array = np.asarray(imu_index_list)
    uwb_time_array = np.asarray(uwb_time_list)
    uwb_index_array = np.asarray(uwb_index_list)

    imu_data = np.frombuffer(imu_buff, dtype=np.float)
    imu_data = imu_data.reshape([-1, 6])

    plt.figure()
    for i in range(3):
        plt.plot(imu_data[:,i],label=str(i))
    plt.title('acc')
    plt.legend()
    plt.grid()


    plt.figure()
    for i in range(3):
        plt.plot(imu_data[:,i+3],label=str(i))
    plt.title('gyr')
    plt.legend()
    plt.grid()



    plt.figure()
    # plt.plot(time_array,'.',label='all')
    # plt.plot()

    plt.plot(imu_index_array, imu_time_array, '.', label='imu')
    plt.plot(uwb_index_array, uwb_time_array, '.', label='uwb')
    plt.legend()
    plt.grid()

    plt.figure()

    plt.plot(imu_index_array[1:], (imu_time_array[1:] - imu_time_array[:-1]), '.', label='imu')
    # plt.plot(uwb_index_array[1:], uwb_time_array[1:] - uwb_time_array[:-1], label='uwb')

    plt.grid()
    plt.legend()

    plt.plot()
    plt.show()
