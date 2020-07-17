# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:47:35 2018

@author: YU-HAN
"""
import matplotlib
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
        p = np.array([lr_target[i,1], lr_target[i,2], lr_target[i,3], 1]).reshape((4,1))
        p = np.dot(T, p)
        lr_target_transformed[i][1:4] = p[0:3].reshape((3,))
    return lr_target_transformed

def cartesian_to_spherical(lr_target):
    lr_target_transformed = lr_target.copy()
    for i in range(0, len(lr_target)):
        p = np.array([lr_target[i,1], lr_target[i,2], lr_target[i,3]]).reshape((3,1))
        
        r = np.linalg.norm(p)
        azimuth = np.arctan2(p[1], p[0])
        azimuth = np.rad2deg(azimuth)
        
        d = np.linalg.norm(p[0:2])
        elevation = np.arctan2(p[2], d)
        elevation = np.rad2deg(elevation)
        lr_target_transformed[i][1:4] = np.array([r, azimuth, elevation])
    return lr_target_transformed


if __name__=='__main__':
    #read data
    file = open('all.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0    
    for row in csvCursor:
        if i==0:
            row = [float(x) for x in row[0:7]]
            A = np.array(row[0:7])
        elif i>0:
            row = [float(x) for x in row[0:7]]
            A = np.row_stack((A, row))
        i=i+1
    file.close()

    label = [18, 1, 2, 3, 7, 8, 9, 12, 13, 14, 15, 16, 17, 20, 22, 26, 27]
    data_t = A[np.isin(A[:,0], label)]

    original = cartesian_to_spherical(data_t)


## RCS
    T = transform_matrix(0.146602, 0.197916, -0.213417, 32.59, 1.9, -0.3)
    transformed_data_t = transform_points(data_t, np.linalg.inv(T))
    rcs = cartesian_to_spherical(transformed_data_t)

## Select
    T = transform_matrix(0.149781, 0.193565, -0.21, 32.5908, 1.17012, 0.17)
    transformed_data_t = transform_points(data_t, np.linalg.inv(T))
    select = cartesian_to_spherical(transformed_data_t)

    
    transformed_data_t = transform_points(data_t, np.linalg.inv(T))
    s = cartesian_to_spherical(transformed_data_t)


    plt.figure()
    plt.scatter(rcs[:,3], rcs[:,6], s = 3, c = s[:,0], cmap = 'rainbow')
    plt.savefig('ranbow.png', dpi = 1000)
    plt.show()

    plt.figure()
    plt.scatter(original[:,3], original[:,6], s = 3, color = 'green', label = 'Before Optimization')
    plt.scatter(rcs[:,3], rcs[:,6], s = 3, color = 'red', label = 'After Optimization')
    plt.legend(loc='upper right', fontsize = 8)
    plt.savefig('comparison.png', dpi = 1000)
    plt.show()

# RCS
    plt.figure()
    c0 = 5.64
    c2 = -0.3
    x = np.arange(-10, 10, 0.1)    
    sigma = c2*(x**2) + c0
    sigma[sigma[:] < -10] = -10
    plt.plot(x, sigma, color = 'black')
    plt.scatter(rcs[:,3], rcs[:,6], s = 3, color = 'red', label = 'After Optimization')
    plt.savefig('curve.png', dpi = 1000)
    plt.show()

# Select
    plt.figure()
    plt.scatter(select[:,3], select[:,6], s = 3, color = 'blue', label = 'After Optimization')
    line3_x = np.zeros((1,2))
    line4_x = np.zeros((1,2))
    line3_y = np.zeros((1,2))
    line4_y = np.zeros((1,2))
    line3_x[0,0] = -15
    line3_y[0,0] = 12
    line3_x[0,1] = 10
    line3_y[0,1] = 12

    line4_x[0,0] = -15
    line4_y[0,0] = -9.5
    line4_x[0,1] = 10
    line4_y[0,1] = -9.5
    plt.plot(line3_x[0,:], line3_y[0,:], color = 'red', linestyle="-")
    plt.plot(line4_x[0,:], line4_y[0,:], color = 'purple', linestyle="-")
    plt.axhspan(12, 22, facecolor='red', alpha=0.1)
    plt.axhspan(-11, -9.5, facecolor='purple', alpha=0.1)

    line1_x = np.zeros((1,2))
    line1_y = np.zeros((1,2))
    line1_x[0,0] = 0
    line1_y[0,0] = -11
    line1_x[0,1] = 0
    line1_y[0,1] = 25
    plt.plot(line1_x[0,:], line1_y[0,:], color = 'black', linestyle="--")

    # plt.legend(loc='upper right', fontsize = 8)
    plt.xlim([-12, 12])
    plt.ylim([-11, 22])
    
    plt.xlabel('elevation angle ψ[deg]')
    plt.ylabel('RCS strength value δ')
    plt.savefig('span.png', dpi = 1000)
    plt.show()

    