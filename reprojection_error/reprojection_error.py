# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:22:18 2019

@author: User
"""
import matplotlib.pyplot as plt
import csv
import numpy as np
import math

def transform_matrix(tx, ty, tz, yaw, pitch, roll):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    translation = np.array((tx, ty, tz)).reshape((3,1))
    
    yawMatrix = np.matrix([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
    pitchMatrix = np.matrix([[math.cos(pitch), 0, math.sin(pitch)],[0, 1, 0],[-math.sin(pitch), 0, math.cos(pitch)]])
    rollMatrix = np.matrix([[1, 0, 0],[0, math.cos(roll), -math.sin(roll)],[0, math.sin(roll), math.cos(roll)]])

    R = yawMatrix * pitchMatrix * rollMatrix
    
    T = np.concatenate((R, translation), axis=1)
    T = np.row_stack((T, [0,0,0,1]))
    
    return T

def transform_points(lr_target, T):
    lr_target_transformed = lr_target.copy()
    for i in range(0, len(lr_target)):
        p = np.array([lr_target[i,0], lr_target[i,1], lr_target[i,2], 1]).reshape((4,1))
        p = np.dot(T, p)
        lr_target_transformed[i][0:3] = p[0:3].reshape((3,))
    return lr_target_transformed

if __name__=='__main__':
    #read data
    file = open('correspondences.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:7]]
            A = np.array(row[0:7])
        elif i>1:
            row = [float(x) for x in row[0:7]]
            A = np.row_stack((A, row))
        i=i+1
    A = A[np.lexsort(A.T)]
    file.close()
    

    # After Reprojection-Error Optimization
    T = transform_matrix(0.17318, 0.199038, -1.465203, 33.3701, -1.87829, 1.80774)
    transformed_points = transform_points(A[:, 1:4], np.linalg.inv(T))

    lidar_norm=np.linalg.norm(transformed_points, axis=1)
    lidar_azimuth = np.arctan2(transformed_points[:,1], transformed_points[:,0]) * 180 / np.pi
    diff_x = lidar_norm * np.cos(np.deg2rad(lidar_azimuth)) - A[:,4] * np.cos(np.deg2rad(A[:,5]))
    diff_y = lidar_norm * np.sin(np.deg2rad(lidar_azimuth)) - A[:,4] * np.sin(np.deg2rad(A[:,5]))
    square_root_1 = np.sqrt(np.square(diff_x) + np.square(diff_y))
    print('Mean_reproj: ', np.average(square_root_1))
    print('Var_reproj:', np.var(square_root_1))

    # After RCS Optimization
    T = transform_matrix(0.17318, 0.199038, -0.24597, 33.3701, -0.775039, 0.103722)
    transformed_points = transform_points(A[:, 1:4], np.linalg.inv(T))

    lidar_norm=np.linalg.norm(transformed_points, axis=1)
    lidar_azimuth = np.arctan2(transformed_points[:,1], transformed_points[:,0]) * 180 / np.pi
    diff_x = lidar_norm * np.cos(np.deg2rad(lidar_azimuth)) - A[:,4] * np.cos(np.deg2rad(A[:,5]))
    diff_y = lidar_norm * np.sin(np.deg2rad(lidar_azimuth)) - A[:,4] * np.sin(np.deg2rad(A[:,5]))
    square_root_2 = np.sqrt(np.square(diff_x) + np.square(diff_y))
    print('Mean_RCS: ', np.average(square_root_2))
    print('Var_RCS:', np.var(square_root_2))

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    h1 = square_root_1[A[:,0] == 3]
    axs.hist(square_root_1, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7, label = "Reprojection Error Optimization")
    axs.hist(square_root_2, bins=40, normed=0, facecolor="red", edgecolor="black", alpha=0.7, label = "RCS Error Optimization")

    axs.set_xlabel("Euclidean distance [m]")
    axs.set_ylabel("times")
    plt.legend(loc='upper right')
    plt.savefig('reprojection_error.png', dpi = 300)
    plt.xlim(0,)
    plt.show()
    