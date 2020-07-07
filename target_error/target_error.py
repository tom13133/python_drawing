# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:11:39 2019

@author: User
"""


import matplotlib.pyplot as plt
import csv
import numpy as np
import math
if __name__=='__main__':
    #read data
    file = open('lidar_data_raw.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0
    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:len(row)]]
            t_dec = np.array(row[0:len(row)])
        elif i>1:
            row = [float(x) for x in row[0:len(row)]]
            t_dec = np.row_stack((t_dec, row))
        i=i+1
    file.close()

    file = open('target_l.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0
    
    for row in csvCursor:
        if i==0:
            row = [float(x) for x in row[0:len(row)]]
            t_loc = np.array(row[0:len(row)])
        elif i>1:
            row = [float(x) for x in row[0:len(row)]]
            t_loc = np.row_stack((t_loc, row))
        i=i+1
    file.close()

    t_loc = t_loc[t_loc[:,0] == 0]
    t_loc[:,2] = t_loc[:,2]
    t_loc[:,4] = t_loc[:,4]
    diff_t = t_loc[:, 2:5] - t_dec[:, 1:4]
    diff_t_norm = np.linalg.norm(diff_t, axis=1,)
    print('offset:(', np.average(diff_t[:, 0]), ', ', np.average(diff_t[:, 1]), ', ', np.average(diff_t[:, 2]), ')')
    print('var:(', np.var(diff_t[:, 0]), ', ', np.var(diff_t[:, 1]), ', ', np.var(diff_t[:, 2]), ')')
    print('Average distance:', np.average(diff_t_norm))
    a = np.array(diff_t_norm <= 0.05, dtype='bool')
    count = 0
    for tmp in a:
        if tmp == True:
            count += 1
    percent = count/a.size
    print('Percentage:', percent)
    plt.figure()
    plt.hist(diff_t_norm, density=0, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.xlabel("Euclidean distance [m]")
    plt.ylabel("times")
    plt.xlim(0,)
    plt.savefig("Evaluation.png", dpi = 300)
    plt.show()