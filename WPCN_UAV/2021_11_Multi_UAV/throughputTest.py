import sys

import helper as h_
import formula as f
import algorithms as algo

import math
import random
import copy
import numpy as np
from shapely.geometry import LineString

import tensorflow as tf
import time

timeCheck = h_.loadSettings({'timeCheck':'logical'})['timeCheck']

# paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1

# wkl, ql(t) : wkl = (xkl, ykl, 0), ql(t) = (xl(t), yl(t), hl(t)), h_min <= hl(t) <= h_max
#              where w = [[l, k, xkl, ykl, 0], ...]
#                and q = [[l, t, xlt, ylt, hlt], ...]

def convertToNumeric(value):
    try:
        return float(value)
    except:
        return float(int(value))

def throughputTest(M, T, L, devices, width, height,
                   H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_, S_,
                   deviceName, QTable_rate):

    # create list of devices (randomly place devices)
    deviceList = []

    for i in range(devices):
        xVal = random.random() * width
        yVal = random.random() * height
        device_i = [xVal, yVal]
        deviceList.append(device_i)

    # clustering
    (q, w) = algo.kMeansClustering(L, deviceList, width, height, H, T, False, True)

    # save device info
    
    return 0

if __name__ == '__main__':

    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    warnings.simplefilter('ignore')

    # load settings
    paperArgs = h_.loadSettings({'deviceName':'str',
                                'QTable_rate':'float',
                                'fc':'float', 'B':'float', 'o2':'float',
                                'b1':'float', 'b2':'float',
                                'alpha':'float',
                                'u1':'float', 'u2':'float',
                                'S_':'float',
                                'alphaL':'float',
                                'r_':'float',
                                'width':'float', 'height':'float',
                                'M':'int', 'L':'int', 'devices':'int', 'T':'int',
                                'H':'float'})

    deviceName = paperArgs['deviceName']
    QTable_rate = paperArgs['QTable_rate']
    fc = paperArgs['fc']
    B = paperArgs['B']
    o2 = paperArgs['o2']
    b1 = paperArgs['b1']
    b2 = paperArgs['b2']
    alpha = paperArgs['alpha']
    u1 = paperArgs['u1']
    u2 = paperArgs['u2']
    S_ = paperArgs['S_']
    alphaL = paperArgs['alphaL']
    r_ = paperArgs['r_']
    width = paperArgs['width']
    height = paperArgs['height']
    M = paperArgs['M']
    L = paperArgs['L']
    devices = paperArgs['devices']
    T = paperArgs['T']
    H = paperArgs['H']

    # write file for configuration info
    configFile = open('config_info.txt', 'w')
    
    configContent = 'deviceName=' + str(deviceName) + '\n'
    configContent += 'QTable_rate=' + str(QTable_rate) + '\n'
    configContent += 'fc=' + str(fc) + '\n'
    configContent += 'B=' + str(B) + '\n'
    configContent += 'o2=' + str(o2) + '\n'
    configContent += 'b1=' + str(b1) + '\n'
    configContent += 'b2=' + str(b2) + '\n'
    configContent += 'alpha=' + str(alpha) + '\n'
    configContent += 'u1=' + str(u1) + '\n'
    configContent += 'u2=' + str(u2) + '\n'
    configContent += 'S_=' + str(S_) + '\n'
    configContent += 'alphaL=' + str(alphaL) + '\n'
    configContent += 'r_=' + str(r_) + '\n'
    configContent += 'width=' + str(width) + '\n'
    configContent += 'height=' + str(height) + '\n'
    configContent += 'M=' + str(M) + '\n'
    configContent += 'L=' + str(L) + '\n'
    configContent += 'devices=' + str(devices) + '\n'
    configContent += 'T=' + str(T) + '\n'
    configContent += 'H=' + str(H)

    configFile.write(configContent)
    configFile.close()

    # run throughput test
    throughputTest(M, T, L, devices, width, height,
                   H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_, S_,
                   deviceName, QTable_rate)
