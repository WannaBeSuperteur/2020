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

# move UAV using direction info
#  x       mod 3 == 0 -> turn left, 1 -> straight, 2 -> turn right
# (x // 3) mod 3 == 0 -> forward  , 1 -> hold    , 2 -> backward
# (x // 9) mod 3 == 0 -> h+1      , 1 -> h       , 2 -> h-1

# initial direction: positive direction of x-axis (0 degree)
# q = [[l, t, xlt, ylt, hlt], ...]
def moveUAV(q, directionList, N, L, width, height):

    for l in range(L):
        degree = 0.0
        
        for t in range(N):

            direction = directionList[l * N + t]

            turn =  direction       % 3 # turn left, straight or turn right
            fb   = (direction // 3) % 3 # forward, hold or backward
            ud   = (direction // 9) % 3 # h+1, h or h-1
            print(l, t, turn, fb, ud)

            currentLocation = q[l * (N+1) + t][2:] # [xlt, ylt, hlt] of the UAV

            # initialize new X, Y and H
            new_X = currentLocation[0]
            new_Y = currentLocation[1]
            new_H = currentLocation[2]

            # turn left, straight or turn right
            if turn == 0:
                degree += 45
            elif turn == 2:
                degree -= 45

            # forward, hold or backward
            if fb == 0:
                new_X += 5 * math.cos(degree * math.pi / 180)
                new_Y += 5 * math.sin(degree * math.pi / 180)
            elif fb == 2:
                new_X -= 5 * math.cos(degree * math.pi / 180)
                new_Y -= 5 * math.sin(degree * math.pi / 180)

            new_X = np.clip(new_X, 0.0, width)
            new_Y = np.clip(new_Y, 0.0, height)

            # height
            if ud == 0:
                new_H += 1
            elif ud == 2:
                new_H -= 1

            if new_H < 0: new_H = 0.0

            # update new q
            q[l * (N+1) + t+1] = [l, t+1, new_X, new_Y, new_H]

def throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU):

    # create list of devices (randomly place devices)
    deviceList = []

    for i in range(devices):
        xVal = random.random() * width
        yVal = random.random() * height
        device_i = [xVal, yVal]
        deviceList.append(device_i)

    # clustering
    (q, w) = algo.kMeansClustering(L, deviceList, width, height, H, N, False, True)

    # save device info
    print('< q >')
    print(np.array(q))

    print('\n< w >')
    print(np.array(w))

    #### TEST ####
    
    # make direction list using random
    directionList = []
    for i in range(N * L):
        directionList.append(random.randint(0, 26))

    # move UAV from time from 0 to T (N+1 times, N moves), for all UAVs of all clusters
    moveUAV(q, directionList, N, L, width, height)

    print('< q >')
    print(np.round_(q, 6))

    # compute common throughput using q and directionList
    # update alkl for each time from 0 to T (N+1 times, N moves)
    alkl = []

    # fill alkl first (iteration: L0 * K1 * L1 * T)
    # if L0 = L1 = 5, K1 = 10 per cluster, T = 30,
    # total iteration count is 5 * 10 * 5 * 30 = 7500
    for i in range(L):
        for j in range(L):

            # the number of devices in cluster j
            start_index = f.find_wkl(w, 0, j)
            end_index   = f.find_wkl(w, 0, j+1)
            devices     = end_index - start_index
        
            for k in range(devices):
                for t in range(N):
                    alkl.append([i, k, j, t, 0])

    # al : al[0] for each UAV l
    al = []
    for l in range(L): al.append(algo.transferEnergy(l, l))

    # speed of light
    c = 300000000

    # throughput results
    throughputs = []
    
    for l in range(L):

        # the number of devices in cluster l
        start_index = f.find_wkl(w, 0, l)
        end_index   = f.find_wkl(w, 0, l+1)
        devices     = end_index - start_index
        
        for t in range(N):
            print(str(l) + '/' + str(L) + ', ' + str(t) + '/' + str(N))

            # decide the device to communicate with
            deviceToCommunicate = algo.findDeviceToCommunicate(q, w, l, t, N, s, b1, b2,
                                                               mu1, mu2, fc, c, alphaL)

            # update alkl, in the form of [[l0, k, l1, n, value], ...]
            alkl_index = f.find_alkl(alkl, l, deviceToCommunicate, l, t)
            
            # DO NOT have alkl with (l, deviceToCommunicate, l, t)
            if alkl_index == None or alkl_index == len(alkl):
                alkl.append([l, deviceToCommunicate, l, t, 1])

                # set alkl as 0 for other devices
                for k in range(devices):
                    if k == deviceToCommunicate: continue
                    alkl.append([l, k, l, t, 0])

            # have alkl with (l, deviceToCommunicate, l, t)
            else:
                alkl[alkl_index][4] = 1

                # set alkl as 0 for other devices
                for k in range(devices):
                    if k == deviceToCommunicate: continue
                    alkl_index_k = f.find_alkl(alkl, l, k, l, t)
                    alkl[alkl_index_k] = [l, k, l, t, 0]

            # sort alkl array
            alkl.sort(key=lambda x:x[3])
            alkl.sort(key=lambda x:x[2])
            alkl.sort(key=lambda x:x[1])
            alkl.sort(key=lambda x:x[0])

        # compute average throughput for each device in L
        for k in range(devices):
            thrput = f.formula_11(q, w, l, k, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, alkl, PU)
            throughputs.append(thrput)

        # print average throughput result for each UAV
        print('UAV ' + str(l) + ' thrputs: ' + str(np.round_(throughputs, 6)))

if __name__ == '__main__':

    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    warnings.simplefilter('ignore')

    # load settings
    paperArgs = h_.loadSettings({'fc':'float', 'ng':'float', 'B':'float', 'o2':'float',
                                'b1':'float', 'b2':'float',
                                'alphaP':'float', 'alphaL':'float',
                                'mu1':'float', 'mu2':'float',
                                's':'float', 'PD':'float', 'PU':'float',
                                'width':'float', 'height':'float',
                                'M':'int', 'L':'int', 'devices':'int', 'T':'float', 'N':'int', 'H':'float'})

    fc = paperArgs['fc']
    ng = paperArgs['ng']
    B = paperArgs['B']
    o2 = paperArgs['o2']
    b1 = paperArgs['b1']
    b2 = paperArgs['b2']
    #alphaP = paperArgs['alphaP']
    alphaL = paperArgs['alphaL']
    mu1 = paperArgs['mu1']
    mu2 = paperArgs['mu2']
    s = paperArgs['s']
    #PD = paperArgs['PD']
    PU = paperArgs['PU']
    width = paperArgs['width']
    height = paperArgs['height']
    M = paperArgs['M']
    L = paperArgs['L']
    devices = paperArgs['devices']
    T = paperArgs['T']
    N = paperArgs['N']
    H = paperArgs['H']

    # (from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1)
    # SN = (1 − alphaP)*T/N denotes the length of each subslot
    # for uplink, alphaP stands for the proportion of downlink WPT in a period.

    # Specifically, the 0-th time slot is assigned to the downlink
    # WPT and the n-th time slot, n in N = {1,2,...,N} is allocated to the uplink WIT.
    alphaP = 1/(N+1)

    # write file for configuration info
    configFile = open('config_info.txt', 'w')
    
    configContent = 'fc=' + str(fc) + '\n'
    configContent += 'ng=' + str(ng) + '\n'
    configContent += 'B=' + str(B) + '\n'
    configContent += 'o2=' + str(o2) + '\n'
    configContent += 'b1=' + str(b1) + '\n'
    configContent += 'b2=' + str(b2) + '\n'
    #configContent += 'alphaP=' + str(alphaP) + '\n'
    configContent += 'alphaL=' + str(alphaL) + '\n'
    configContent += 'mu1=' + str(mu1) + '\n'
    configContent += 'mu2=' + str(mu2) + '\n'
    configContent += 's=' + str(s) + '\n'
    #configContent += 'PD=' + str(PD) + '\n'
    configContent += 'PU=' + str(PU) + '\n'
    configContent += 'width=' + str(width) + '\n'
    configContent += 'height=' + str(height) + '\n'
    configContent += 'M=' + str(M) + '\n'
    configContent += 'L=' + str(L) + '\n'
    configContent += 'devices=' + str(devices) + '\n'
    configContent += 'T=' + str(T) + '\n'
    configContent += 'N=' + str(N) + '\n'
    configContent += 'H=' + str(H)

    configFile.write(configContent)
    configFile.close()

    # run throughput test
    #throughputTest(M, T, N, L, devices, width, height, H,
    #               ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU)
    throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, None, alphaL, mu1, mu2, s, None, PU)
