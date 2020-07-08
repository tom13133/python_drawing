# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:16:05 2019

@author: User
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
if __name__=='__main__':
    file_A = open('PEAK.csv', 'r')
    file_B = open('RCS.csv', 'r')
    file_C = open('Select.csv', 'r')
    csvCursor = csv.reader(file_A)
    i = 1    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:6]]
            A = np.array(row[0:6])
        else:
            row = [float(x) for x in row[0:6]]
            A = np.row_stack((A, row))
        i=i+1
    csvCursor = csv.reader(file_B)
    i = 1    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:6]]
            B = np.array(row[0:6])
        else:
            row = [float(x) for x in row[0:6]]
            B = np.row_stack((B, row))
        i=i+1
    csvCursor = csv.reader(file_C)
    i = 1    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:6]]
            C = np.array(row[0:6])
        else:
            row = [float(x) for x in row[0:6]]
            C = np.row_stack((C, row))
        i=i+1
    file_A.close()
    file_B.close()
    file_C.close()

    tf1 = np.zeros((2,3))
    tf1[0,0] = np.mean(A[:,2])
    tf1[0,1] = np.mean(A[:,4])
    tf1[0,2] = np.mean(A[:,5])
    tf1[1,0] = np.var(A[:,2])
    tf1[1,1] = np.var(A[:,4])
    tf1[1,2] = np.var(A[:,5])
    
    tf2 = np.zeros((2,3))
    tf2[0,0] = np.mean(B[:,2])
    tf2[0,1] = np.mean(B[:,4])
    tf2[0,2] = np.mean(B[:,5])
    tf2[1,0] = np.var(B[:,2])
    tf2[1,1] = np.var(B[:,4])
    tf2[1,2] = np.var(B[:,5])
    
    tf3 = np.zeros((2,3))
    tf3[0,0] = np.mean(C[:,2])
    tf3[0,1] = np.mean(C[:,4])
    tf3[0,2] = np.mean(C[:,5])
    tf3[1,0] = np.var(C[:,2])
    tf3[1,1] = np.var(C[:,4])
    tf3[1,2] = np.var(C[:,5])

    print('Proposed')
    for i in range(0, 2):
        for j in range(0, 3):
            if i == 0:
                print(tf1[i, j], ' ', end='')
            else:
                print("{:e}".format(tf1[i, j]), ' ', end='')
        print(end='\n')


    print('RCS')
    for i in range(0, 2):
        for j in range(0, 3):
            if i == 0:
                print(tf2[i, j], ' ', end='')
            else:
                print("{:e}".format(tf2[i, j]), ' ', end='')
        print(end='\n')

    print('Select')
    for i in range(0, 2):
        for j in range(0, 3):
            if i == 0:
                print(tf3[i, j], ' ', end='')
            else:
                print("{:e}".format(tf3[i, j]), ' ', end='')
        print(end='\n')

    
    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)

    axs[0].hist(C[:,2], bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[0].hist(B[:,2], bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[0].hist(A[:,2], bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)


    axs[0].set_xlabel("translation z [m]")
    axs[0].set_ylabel("times")
    
    axs[1].hist(C[:,4], bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[1].hist(B[:,4], bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[1].hist(A[:,4], bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[1].set_xlabel("pitch [°]")
    axs[1].set_ylabel("times")
 

    axs[2].hist(C[:,5], bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[2].hist(B[:,5], bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[2].hist(A[:,5], bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[2].set_xlabel("roll [°]")
    axs[2].set_ylabel("times")

    plt.savefig('RCS_hist.png', dpi = 300)
    plt.show()
