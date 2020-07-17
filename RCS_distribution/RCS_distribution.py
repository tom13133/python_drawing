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
    
    label = [4, 7, 12, 13, 16, 19, 22, 23, 25, 26, 28, 29, 31, 34, 37,  43,  49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91]
    #read data
    file = open('correspondence_92.csv', 'r')
    csvCursor = csv.reader(file)
    i = 0    
    for row in csvCursor:
        if i==1:
            row = [float(x) for x in row[0:7]]
            A = np.array(row[0:7])
        elif i>1:
            row = [float(x) for x in row[0:7]]
            if row[6] == -10:
                continue
            A = np.row_stack((A, row))
        i=i+1
    A = A[np.lexsort(A.T)]
    file.close()
    
    A = A[np.isin(A[:,0], label)]

    line1_x = np.zeros((1,2))
    line2_x = np.zeros((1,2))
    line3_x = np.zeros((1,2))
    line1_y = np.zeros((1,2))
    line2_y = np.zeros((1,2))
    line3_y = np.zeros((1,2))
    
    line1_x[0,1] = 10
    line1_y[0,1] = 0.743
    line2_x[0,1] = 10
    line3_x[0,1] = 10
    line3_y[0,1] = -0.83
    

    data = np.random.randint(0, 255, size=[40, 40, 40])


    T = transform_matrix(0.146602, 0.197916, -0.213417, 32.59, 1.9, -0.3)
    transformed_points = transform_points(A[:, 1:4], np.linalg.inv(T))
    
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.set_ylim([0, 10])
    cm = plt.cm.get_cmap('rainbow')
    ax.scatter(transformed_points[:,1], transformed_points[:,0], transformed_points[:,2], c = A[:,6], cmap = cm)
    plt.savefig('distribution_3d.png', dpi = 300)
    plt.show()

    # Plot distribution
    plt.figure()
    plt.plot(line1_x[0,:], line1_y[0,:], color = 'black', linestyle="--")
    plt.plot(line2_x[0,:], line2_y[0,:], color = 'black', linestyle="-")
    plt.plot(line3_x[0,:], line3_y[0,:], color = 'black', linestyle="--")
    cm = plt.cm.get_cmap('rainbow')
    pts = plt.scatter(transformed_points[:,0], transformed_points[:,2], s = 10, c = A[:,6], vmin=-10, vmax=21, cmap=cm)
    clb = plt.colorbar(pts)
    clb.ax.set_title('RCS δ')

    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.xlim(0, 10)
    plt.savefig('distribution_2d.png', dpi = 300)
    plt.show()