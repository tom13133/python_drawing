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
    
    file = open('tf.csv', 'r')
    csvCursor = csv.reader(file)


    i = 1    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:6]]
            A = np.array(row[0:6])
        elif i == 2:
            row = [float(x) for x in row[0:6]]
            B = np.array(row[0:6])   
        elif i == 3:
            row = [float(x) for x in row[0:6]]
            C = np.array(row[0:6])  
        elif i%3 == 1:
            row = [float(x) for x in row[0:6]]
            A = np.row_stack((A, row))
        elif i%3 == 2:
            row = [float(x) for x in row[0:6]]
            B = np.row_stack((B, row))
        elif i%3 == 0:
            row = [float(x) for x in row[0:6]]
            C = np.row_stack((C, row))
        i=i+1


    tx = A[:,0]
    ty = A[:,1]
    tz = A[:,2]
    yaw = A[:,3]
    pitch = A[:,4]
    roll = A[:,5]
    
    tz_ = B[:,2]
    pitch_ = B[:,4]
    roll_ = B[:,5]

    tx_ = C[:,0]
    ty_ = C[:,1]
    yaw_ = C[:,3]
    
    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)

    axs[0].hist(tx, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[0].hist(tx_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[0].set_xlabel("translation x [m]")
    axs[0].set_ylabel("times")
    
    axs[1].hist(ty, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[1].hist(ty_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[1].set_xlabel("translation y [m]")
    axs[1].set_ylabel("times")
 

    axs[2].hist(yaw, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[2].hist(yaw_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    axs[2].set_xlabel("yaw [°]")
    axs[2].set_ylabel("times")
    
    plt.savefig('Monte_1.png', dpi = 300)
    plt.show()
    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
    
    axs[0].hist(tz, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[0].hist(tz_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[0].set_xlabel("translation z [m]")
    axs[0].set_ylabel("times")


    axs[1].hist(pitch, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[1].hist(pitch_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[1].set_xlabel("pitch [°]")
    axs[1].set_ylabel("times")


    axs[2].hist(roll, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    axs[2].hist(roll_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    axs[2].set_xlabel("roll [°]")
    axs[2].set_ylabel("times")    
    
    plt.savefig('Monte_2.png', dpi = 300)
    plt.show()
    tf1 = np.zeros((2,6))
    tf1[0,0] = np.mean(A[:,0])
    tf1[0,1] = np.mean(A[:,1])
    tf1[0,2] = np.mean(A[:,2])
    tf1[0,3] = np.mean(A[:,3])
    tf1[0,4] = np.mean(A[:,4])
    tf1[0,5] = np.mean(A[:,5])
    tf1[1,0] = np.var(A[:,0])
    tf1[1,1] = np.var(A[:,1])
    tf1[1,2] = np.var(A[:,2])
    tf1[1,3] = np.var(A[:,3])
    tf1[1,4] = np.var(A[:,4])
    tf1[1,5] = np.var(A[:,5])
    
    tf2 = np.zeros((2,6))
    tf2[0,0] = np.mean(B[:,0])
    tf2[0,1] = np.mean(B[:,1])
    tf2[0,2] = np.mean(B[:,2])
    tf2[0,3] = np.mean(B[:,3])
    tf2[0,4] = np.mean(B[:,4])
    tf2[0,5] = np.mean(B[:,5])    
    tf2[1,0] = np.var(B[:,0])
    tf2[1,1] = np.var(B[:,1])
    tf2[1,2] = np.var(B[:,2])
    tf2[1,3] = np.var(B[:,3])
    tf2[1,4] = np.var(B[:,4])
    tf2[1,5] = np.var(B[:,5])    
    
    tf3 = np.zeros((2,6))
    tf3[0,0] = np.mean(C[:,0])
    tf3[0,1] = np.mean(C[:,1])
    tf3[0,2] = np.mean(C[:,2])
    tf3[0,3] = np.mean(C[:,3])
    tf3[0,4] = np.mean(C[:,4])
    tf3[0,5] = np.mean(C[:,5])    
    tf3[1,0] = np.var(C[:,0])
    tf3[1,1] = np.var(C[:,1])
    tf3[1,2] = np.var(C[:,2])
    tf3[1,3] = np.var(C[:,3])
    tf3[1,4] = np.var(C[:,4])
    tf3[1,5] = np.var(C[:,5])    

    print('Reprojection Error Optimization 1st')
    for i in range(0, 2):
        for j in range(0, 6):
            if i == 0:
                print(tf1[i, j], ' ', end='')
            else:
                print("{:e}".format(tf1[i, j]), ' ', end='')
        print(end='\n')
    file.close()
    

    print('RCS Error Optimization')
    for i in range(0, 2):
        for j in range(0, 6):
            if i == 0:
                print(tf2[i, j], ' ', end='')
            else:
                print("{:e}".format(tf2[i, j]), ' ', end='')
        print(end='\n')
    file.close()

    print('Reprojection Error Optimization 2nd')
    for i in range(0, 2):
        for j in range(0, 6):
            if i == 0:
                print(tf3[i, j], ' ', end='')
            else:
                print("{:e}".format(tf3[i, j]), ' ', end='')
        print(end='\n')
    file.close()

'''
    plt.hist(tx, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(tx_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    plt.xlabel("translation x [m]")
    plt.ylabel("times")

#    plt.title("tx-times plot")
    plt.show()
    
    plt.figure()
    plt.hist(ty, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(ty_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    plt.xlabel("translation y [m]")
    plt.ylabel("times")
 
#    plt.title("ty-times plot")
    plt.show()
    
    plt.figure()
    plt.hist(tz, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(tz_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    plt.xlabel("translation z [m]")
    plt.ylabel("times")
#    plt.title("tz-times plot")
    plt.show()
    
    plt.figure()
    plt.hist(yaw, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(yaw_, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    plt.xlabel("yaw [°]")
    plt.ylabel("times")
#    plt.title("yaw-times plot")
    plt.show()
    
    plt.figure()
    plt.hist(pitch, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(pitch_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    plt.xlabel("pitch [°]")
    plt.ylabel("times")
#    plt.title("pitch-times plot")
    plt.show()
    
    plt.figure()
    plt.hist(roll, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(roll_, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    plt.xlabel("roll [°]")
    plt.ylabel("times")
#    plt.title("roll-times plot")
    plt.show()
'''