# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 07:12:09 2020

@author: User
"""


import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# Deg/sec
VELOCITY = np.rad2deg(0.3)
TIME_SPAN = 30
LIDAR_FEQ = 10
RADAR_FEQ = 20
LIDAR_UNCERTAINLY = 0.02
RADAR_ACCURACY_RANGE = 0.25
RADAR_ACCURACY_AZIMUTH = 1
RADAR_BIAS = 0.00
TIMEDELAY = 0.1
BASETIME = 1568713910

# range, azimuth, elevation
# =============================================================================
# target_s = np.array([[25., 45., 0],
#                      [26., 45, 0],
#                      [27., 45, 0],
#                      [28., 45, 0]])
# =============================================================================
# =============================================================================
# target_s = np.array([[9., 45., 0],
#                      [10., 45, 0],
#                      [11., 45, 0],
#                      [12., 45, 0]])
# =============================================================================
# =============================================================================
# target_s = np.array([[13., 45., 0],
#                      [14., 45, 0],
#                      [15., 45, 0],
#                      [16., 45, 0]])
# =============================================================================
# =============================================================================
# target_s = np.array([[17., 45., 0],
#                      [18., 45, 0],
#                      [19., 45, 0],
#                      [20., 45, 0]])
# =============================================================================
target_s = np.array([[5., 30., 0],
                     [10., 15, 0],
                     [15., -15, 0],
                     [20., 0, 0]])
# =============================================================================
# target_s = np.array([[5., 45., 0],
#                      [6., 45, 0],
#                      [7., 45, 0],
#                      [8., 45, 0],
#                      [9., 45., 0],
#                      [10., 45, 0],
#                      [11., 45, 0],
#                      [12., 45, 0],
#                      [13., 45., 0],
#                      [14., 45, 0],
#                      [15., 45, 0],
#                      [16., 45, 0],
#                      [17., 45., 0],
#                      [18., 45, 0],
#                      [19., 45, 0],
#                      [20., 45, 0]])
# =============================================================================



def transform_matrix_XYZ(tx, ty, tz, yaw, pitch, roll):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    translation = np.array((tx, ty, tz)).reshape((3,1))
    
    yawMatrix = np.matrix([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
    pitchMatrix = np.matrix([[math.cos(pitch), 0, math.sin(pitch)],[0, 1, 0],[-math.sin(pitch), 0, math.cos(pitch)]])
    rollMatrix = np.matrix([[1, 0, 0],[0, math.cos(roll), -math.sin(roll)],[0, math.sin(roll), math.cos(roll)]])

    R =  rollMatrix * pitchMatrix * yawMatrix
    
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

def spherical_to_cartesian(lr_target):
    lr_target_transformed = lr_target.copy()
    for i in range(0, len(lr_target)):
        x = lr_target[i, 0] * np.cos(np.deg2rad(lr_target[i,1])) * np.cos(np.deg2rad(lr_target[i,2]))
        y = lr_target[i, 0] * np.sin(np.deg2rad(lr_target[i,1])) * np.cos(np.deg2rad(lr_target[i,2]))
        z = lr_target[i, 0] * np.sin(np.deg2rad(lr_target[i,2]))

        lr_target_transformed[i][0:3] = np.array([x, y, z])
    return lr_target_transformed

def radar_uncertainty(radar_data_s):
    radar_data_noise_s = radar_data_s.copy()
    for i in range(0, len(radar_data_s)):
        tmp = radar_data_s[i]
        r = tmp[0] + np.random.normal(0, RADAR_ACCURACY_RANGE/3, 1)
        a = tmp[1] + np.random.normal(0, RADAR_ACCURACY_AZIMUTH/3, 1)
        radar_data_noise_s[i, 0] = r
        radar_data_noise_s[i, 1] = a
    return radar_data_noise_s

def lidar_uncertainty(lidar_data_c):
    lidar_data_noise_c = lidar_data_c.copy()
    for i in range(0, len(lidar_data_c)):
        lidar_data_noise_c[i, 0] = lidar_data_c[i, 0] + np.random.normal(0, LIDAR_UNCERTAINLY/3, 1)
        lidar_data_noise_c[i, 1] = lidar_data_c[i, 1] + np.random.normal(0, LIDAR_UNCERTAINLY/3, 1)
        lidar_data_noise_c[i, 2] = lidar_data_c[i, 2] + np.random.normal(0, LIDAR_UNCERTAINLY/3, 1)
    return lidar_data_noise_c


T_lr = transform_matrix_XYZ(0.15, 0.2, -0.25, 32.5, 1.4, -1.3)
T_rl = np.linalg.inv(T_lr)

if __name__=='__main__':
    with open('lidar_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
    with open('radar_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
    degree_range = VELOCITY * TIME_SPAN
    samples = TIME_SPAN * RADAR_FEQ
    for label in range(0, len(target_s)):
        direction = 1
        tmp = target_s[label]
        for i in range(0, samples):
            if i==0:
                radar_data = np.array(tmp)
            elif i>0:
                radar_data = np.row_stack((radar_data, tmp))
    
            if tmp[1] >= 45:
                direction = -1
            elif tmp[1] <= -45:
                direction = 1
            tmp = np.array([tmp[0], tmp[1]+VELOCITY/RADAR_FEQ*direction, tmp[2]])

        radar_data_c = spherical_to_cartesian(radar_data)
        lidar_data_c = transform_points(radar_data_c, T_lr)
        radar_data_noise_s = radar_uncertainty(radar_data)
        radar_data_noise_s = np.around(radar_data_noise_s, 1)
        radar_data_noise_c = spherical_to_cartesian(radar_data_noise_s)
        lidar_data_noise_c = lidar_uncertainty(lidar_data_c)
        if label == 0:
            radar_data_noise_s_all = radar_data_noise_s
        else:
            radar_data_noise_s_all = np.row_stack((radar_data_noise_s_all, radar_data_noise_s))
        for i in range(0, samples):
            with open('lidar_data.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([str(label), str(i / RADAR_FEQ + BASETIME), lidar_data_noise_c[i, 0], lidar_data_noise_c[i, 1], lidar_data_noise_c[i, 2]])
            with open('radar_data.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([str(label), str(i / RADAR_FEQ + BASETIME + TIMEDELAY), radar_data_noise_c[i, 0], radar_data_noise_c[i, 1], radar_data_noise_c[i, 2]])

    plt.figure()
    radar_data_noise_c_all = spherical_to_cartesian(radar_data_noise_s_all)
    plt.scatter(radar_data_noise_c_all[:,1], radar_data_noise_c_all[:,0], s = 3, color = 'green')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig('reproj_sim.png', dpi = 300)
    plt.show()