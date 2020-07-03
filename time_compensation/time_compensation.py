# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:51:12 2019

@author: User
"""


import matplotlib.pyplot as plt
import csv
import numpy as np
import math

def transform_matrix():
    yaw = np.deg2rad(33.3701)
    pitch = np.deg2rad(-1.87829)
    roll = np.deg2rad(1.80774)
    translation = np.array((0.17318, 0.199038, -1.465203)).reshape((3,1))

    yawMatrix = np.matrix([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
    pitchMatrix = np.matrix([[math.cos(pitch), 0, math.sin(pitch)],[0, 1, 0],[-math.sin(pitch), 0, math.cos(pitch)]])
    rollMatrix = np.matrix([[1, 0, 0],[0, math.cos(roll), -math.sin(roll)],[0, math.sin(roll), math.cos(roll)]])

    R = yawMatrix * pitchMatrix * rollMatrix
    
    T = np.concatenate((R, translation), axis=1)
    T = np.row_stack((T, [0,0,0,1]))
    
    return T
    
def transform_points(lr_target):
    T = transform_matrix()
    lr_target_transformed = lr_target.copy()
    for i in range(0, len(lr_target)):
        p = np.array([lr_target[i,1], lr_target[i,2], lr_target[i,3], 1]).reshape((4,1))
        p = np.dot(np.linalg.inv(T), p)
        lr_target_transformed[i][1:4] = p[0:3].reshape((3,))
    return lr_target_transformed

def cartesian_to_spherical(p):
    r = np.linalg.norm(p[1:4])
    azimuth = np.arctan2(p[2], p[1])
    azimuth = np.rad2deg(azimuth)
    return np.array([p[0], azimuth, r])

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

    label = 2
    r_data_t = r_data[r_data[:,0] == label]
    l_data_t = l_data[l_data[:,0] == label]
    r_data_t_ = r_data_t[r_data_t[:,5] == 1]
    r_data_target = r_data_t[:, 1:6]
    r_data_target_ = r_data_t_[:, 1:6]
    l_data_target = l_data_t[:, 1:5]
        
    l_data_transformed = transform_points(l_data_target)

    for i in range(0, len(l_data_transformed)):
        if i == 0:
            s = cartesian_to_spherical(l_data_transformed[i][0:4])
        else:
            temp = cartesian_to_spherical(l_data_transformed[i][0:4])
            s = np.row_stack((s, temp))

    time_delay = -0.05

    x_r = r_data_target[:,1];
    y_r = r_data_target[:,2];
    a_r = np.rad2deg(np.arctan2(y_r, x_r))

    x_r_ = r_data_target_[:,1];
    y_r_ = r_data_target_[:,2];
    a_r_ = np.rad2deg(np.arctan2(y_r_, x_r_))
    tv_r = r_data_target_[:,0]

    t_r = r_data_target[:,0] -r_data_target[0,0] 
    t_r_ = r_data_target[:,0] - r_data_target[0,0] + time_delay
    x_l = np.multiply(s[:,2], np.cos(s[:,1]*math.pi/180));
    y_l = np.multiply(s[:,2], np.sin(s[:,1]*math.pi/180));
    a_l = np.rad2deg(np.arctan2(y_l,x_l))
    t_l = s[:,0] - r_data_target[0,0]

    region_x = np.array([52, 52, 52.5, 52.5, 52])
    region_y = np.array([0, 20, 20, 0, 0])

    # time v.s. azimuth
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(t_r, a_r, s = 30, color = 'green', label = 'radar data')
    plt.plot(t_l, a_l, linewidth=1.5, color = 'black', linestyle="-", label = 'lidar data')
    plt.plot(region_x, region_y, linewidth=1.5, color = 'red', linestyle="--")
    plt.xlabel('time')
    plt.ylabel('azimuth(deg)')
    plt.legend(loc='upper right')
    scale = (t_r[t_r.shape[0]-1] - t_r[0])/8

    plt.xlim(51.999, 52.5)
    plt.ylim(0, 20.1)

    plt.subplot(1,2,2)
    plt.scatter(t_r_, a_r, s = 30, color = 'blue', label = 'radar data')
    plt.plot(t_l, a_l, linewidth=1.5, color = 'black', linestyle="-", label = 'lidar data')
    plt.plot(region_x, region_y, linewidth=1.5, color = 'red', linestyle="--")
    plt.xlabel('time')
    plt.ylabel('azimuth(deg)')
    plt.legend(loc='upper right')
    scale = (t_r[t_r.shape[0]-1] - t_r[0])/8

    plt.xlim(51.999, 52.5)
    plt.ylim(0, 20.1)

    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

    plt.savefig('time_compensation.png', dpi = 700)
    plt.show()

    
