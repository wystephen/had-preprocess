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
import  numpy as np
import scipy as sp


if __name__ == '__main__':
    file_lines = open('/home/steve/Data/HardIU/UN1_Data.log').readlines()

    time_array = np.zeros([len(file_lines),1])
    # for line in file_lines:
    for index in range(len(file_lines)):
        line = file_lines[index]
        time_array[index,0] = line.split('[')[1].split(']')[0]

    plt.figure()
    plt.plot(time_array)
    plt.grid()
    plt.show()

