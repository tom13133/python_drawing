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
    file = open('correspondence_92.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0
    for row in csvCursor:
        if i==0:
            row = [float(x) for x in row[0:7]]
            if row[6] == -10:
                continue
            A = np.array(row[0:7])
        elif i>0:
            row = [float(x) for x in row[0:7]]
            if row[6] == -10:
                continue
            A = np.row_stack((A, row))
        i=i+1
    file.close()


    label = [4, 7, 12, 13, 16, 19, 22, 23, 25, 26, 28, 29, 31, 34, 37,  43,  49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91]

    data_t = A[np.isin(A[:,0], label)]


## Proposed
    T = transform_matrix(0.146602, 0.197916, -0.213417, 32.59, 1.9, -0.3)
    transformed_data_t = transform_points(data_t, np.linalg.inv(T))
    rcs = cartesian_to_spherical(transformed_data_t)

    plt.scatter(rcs[:,3], rcs[:,6], s = 3, color = 'green', label = 'After optimization')

    cm = plt.cm.get_cmap('rainbow')

    c = np.array([[91, 9.19201, -1.27387],
                  [88, 11.2578, -0.659828],
                  [85, 11.4739, -0.720456],
                  [82, 6.51801, -1.06448],
                  [79, 11.9269, -1.12646],
                  [76, 11.7599, -1.14076],
                  [73, 12.5236, -1.23292],
                  [70, 12.8997, -1.198],
                  [67, 13.1204, -1.24931],
                  [64, 13.5416, -1.19934],
                  [61, 12.6149, -1.17719],
                  [22, 13.7995, -1.48032],
                  [19, 14.4615, -1.16332],
                  [16, 15.1524, -1.38815],
                  [13, 12.4398, -1.39746],
                  [12, 7.95005, -0.510934],
                  [10, 6.71401, -0.263278],
                  [26, 7.29354, -0.926315],
                  [25, 10.3501, -1.43956],
                  [49, 12.2124, -1.09364],
                  [1, 7.53368, -0.199015],
                  [23, 7.94694, -0.783843],
                  [4, 20.2567, -1.27935],
                  [28, 10.1979, -1.35442],
                  [52, 10.7868, -0.747485],
                  [7, 17.0903, -1.23836],
                  [55, 12.9306, -1.1917],
                  [29, 11.0534, -1.27087],
                  [31, 13.8738, -1.12633],
                  [34, 16.4229, -1.07512],
                  [37, 14.7261, -1.20483],
                  [40, 9.20095, -0.267097],
                  [43, 15.8518, -1.25769],
                  [46, 6.48132, -0.153819],
                  [58, 12.0666, -1.19743]])

    for i in range(0, len(c)):
        if c[i][0] in label:
            x = np.arange(-10, 10, 0.1)    
            sigma = c[i][2]*(x**2) + c[i][1]
            plt.plot(x, sigma, color = 'red')

    line1_x = np.zeros((1,2))
    line1_y = np.zeros((1,2))
    line1_x[0,0] = 0
    line1_y[0,0] = -11
    line1_x[0,1] = 0
    line1_y[0,1] = 25
    plt.plot(line1_x[0,:], line1_y[0,:], color = 'black', linestyle="--")

    plt.legend(loc='upper right', fontsize = 8)
    plt.xlim([-12, 12])
    plt.ylim([-10, 22])
    
    plt.xlabel('elevation angle ψ[deg]')
    plt.ylabel('RCS strength value δ')
    plt.savefig('curve_azimuth.png', dpi = 1000)
    plt.show()

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    ax.scatter(transformed_data_t[:, 1], rcs[:, 3], rcs[:,6], c = transformed_data_t[:,0], cmap = cm)

    for i in range(0, len(c)):
        if c[i][0] in label:

            z = np.arange(-10, 10, 0.01)
            idx = np.where(transformed_data_t[:,0] == c[i][0])
            x = z.copy()
            x[:] = transformed_data_t[idx[0][5]][1]
            sigma = c[i][2]*(z**2) + c[i][1]
            idx = np.where(sigma < -10)
            x = np.delete(x, idx)
            z = np.delete(z, idx)
            sigma = np.delete(sigma, idx)
            print(c[i][0], ', ', x[0])
            ax.scatter(x, z, sigma, s = 1, color = 'black')


    ax.set_xlabel('X')
    ax.set_ylabel('elevation')
    ax.set_zlabel('RCS')
    ax.set_zlim([-10, 22])
    plt.savefig('curve_3d.png', dpi = 1000)
    plt.show()