# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:51:12 2019

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

def cartesian_to_spherical(lr_target):
    lr_target_transformed = lr_target.copy()
    for i in range(0, len(lr_target)):
        p = np.array([lr_target[i,0], lr_target[i,1], lr_target[i,2]]).reshape((3,1))
        
        r = np.linalg.norm(p)
        azimuth = np.arctan2(p[1], p[0])
        azimuth = np.rad2deg(azimuth)
        
        d = np.linalg.norm(p[0:2])
        elevation = np.arctan2(p[2], d)
        elevation = np.rad2deg(elevation)
        lr_target_transformed[i][0:3] = np.array([r, azimuth, elevation])
    return lr_target_transformed

if __name__=='__main__':
    #read data
    file_r = open('target_r.csv', 'r')
    file_l = open('target_l.csv', 'r')
    csvCursor_r = csv.reader(file_r)
    csvCursor_l = csv.reader(file_l)
    i = 0    
    for row in csvCursor_r:
        if i==1:
            row = [float(x) for x in row[0:6]]
            r_data = np.array(row[0:6])
        elif i>1:
            row = [float(x) for x in row[0:6]]
            r_data = np.row_stack((r_data, row[0:6]))
        i=i+1
    i = 0
    for row in csvCursor_l:
        if i==1:
            row = [float(x) for x in row[0:5]]
            l_data = np.array(row[0:5])
        elif i>1:
            row = [float(x) for x in row[0:5]]
            l_data = np.row_stack((l_data, row[0:5]))
        i=i+1



    label = 3
    r_data_t = r_data[r_data[:,0] == label]
    l_data_t = l_data[l_data[:,0] == label]

    T_lr = transform_matrix(0.141093, 0.335704, -1.36073, 32.1548, -5.85019, 10.4136)
    transformed_points = transform_points(l_data_t[:, 2:5], np.linalg.inv(T_lr))
    s = cartesian_to_spherical(transformed_points)

    time_delay = -0.1

    x_r = r_data_t[:,2];
    y_r = r_data_t[:,3];
    a_r = np.rad2deg(np.arctan2(y_r, x_r))

    t_r = r_data_t[:,1] - r_data_t[0,1]
    t_r_ = r_data_t[:,1] - r_data_t[0,1] + time_delay

    x_l = np.multiply(s[:,0], np.cos(s[:,1]*math.pi/180));
    y_l = np.multiply(s[:,0], np.sin(s[:,1]*math.pi/180));
    a_l = np.rad2deg(np.arctan2(y_l,x_l))
    t_l = l_data_t[:,1] - r_data_t[0,1]

    # region_x = np.array([52, 52, 52.5, 52.5, 52])
    # region_y = np.array([0, 20, 20, 0, 0])

    # time v.s. azimuth
    plt.figure()

    plt.scatter(t_r, a_r, s = 5, color = 'green', label = 'radar data')
    plt.scatter(t_r_, a_r, s = 5, color = 'blue', label = 'radar data (shifted)')
    # plt.plot(region_x, region_y, linewidth=1.5, color = 'red', linestyle="--")
    plt.plot(t_l, a_l, linewidth=1.5, color = 'black', linestyle="-", label = 'lidar data')
    plt.xlabel('time')
    plt.ylabel('azimuth(deg)')
    plt.legend(loc='upper right')
    scale = (t_r[t_r.shape[0]-1] - t_r[0])/8

    plt.xlim(28, 39)
    # plt.ylim(0, 20.1)
    plt.savefig('time_shift.png', dpi = 700)
    plt.show()




    
