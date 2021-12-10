import sys

import helper as h_
import formula as f
import algorithms as algo

import math
import random
import copy
import numpy as np
import pandas as pd
import seaborn as sns

from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import tensorflow as tf
import time

timeCheck = h_.loadSettings({'timeCheck':'logical'})['timeCheck']
printDetails = h_.loadSettings({'printDetails':'logical'})['printDetails']

# note:
# to ensure that MINIMUM THROUGHPUT > 0
# --->           sum of a_l,kl[n] > 0 for all devices

# because Sum(k_l in K_l) a_l,kl[n] <= 1 ... (5),
# it means for each time, only ONE of devices in cluster L can satisfy a_l,kl[n] > 0
# for all devices in cluster L, at least one time slot T, must satisfy a_l,kl[n] > 0

# for each cluster, (N is the number of time slots)
# if      N >  (devices), possible and at least one devices get >=2 chances to satisfy a_l,kl[n] > 0
# else if N == (devices), all devices must get 1 chance to satisfy a_l,kl[n] > 0
# else if N <  (devices), IMPOSSIBLE (minimum throughput is ALWAYS 0)

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

# change color
def changeColor(colorCode, k):
    r = colorCode[1:3]
    g = colorCode[3:5]
    b = colorCode[5:7]

    result = '#'
	
    for x in [r, g, b]:
        x = int(x, 16)
        x = min(int(x * k), 255)
        x = hex(x)
        
        result += '0'*(2 - len(x[2:])) + x[2:]
        
    return result

# make training dataset
# input       : device location (w[:, 2], w[:, 3]) + (height of UAV) + action (dif of x, y, and h of UAV)
# output      : function of throughput change
# action_list : directionList[l*N : l*(N + 1)]

# cluster No. : w[:, 0]
# throughputs : throughputs at time slot t = 0 ... N-1
def modifyArr(arr, y, x, value):
    if y >= 0 and y < len(arr) and x >= 0 and x < len(arr[0]):
        arr[y][x] = value
    
def makeTrainDataset(w, l, action_list, throughputs, t, q):

    w = np.array(w)
    throughputs = np.array(throughputs).T

    # filter for cluster l
    dev_x = []
    dev_y = []
    
    for i in range(len(w)):
        if w[i][0] == l:
            dev_x.append(w[i][2])
            dev_y.append(w[i][3])

    # the number of devices in cluster l
    dev_in_l = len(dev_x)

    # board configuration
    width  = 100
    height = 100

    print('\n [l=' + str(l) + ' t=' + str(t) + '] < dev_x >')
    print(list(dev_x))
    print(' < dev_y >')
    print(list(dev_y))
    print(' < action_list >')
    print(list(action_list))

    thrput       = throughputs[t]
    thrput_after = throughputs[t+1]

    print(' < thrput >')
    print(list(thrput))
    print(' < thrput_after >')
    print(list(thrput_after))

    q_current = q[l * (N+1) + t    ][2:5]
    q_after   = q[l * (N+1) + (t+1)][2:5]

    print(' < q_current >')
    print(q_current)
    print(' < q_after >')
    print(q_after)

    # find the range of the board (unit: width=0.5, height=0.5)
    board = np.zeros((2*width, 2*height))

    # max device and min device
    maxThrput = max(thrput)
    minThrput = min(thrput)

    # normalized throughput
    thrput_   = [(thrput[i] - minThrput) / (maxThrput - minThrput) for i in range(len(thrput))]

    for i in range(dev_in_l):

        # place the device on the board (mark as [[0.5, 1.0, 0.5], [1.0, 1.2, 1.0], [0.5, 1.0, 0.5]])
        board_x = int(dev_x[i] * 2)
        board_y = int(dev_y[i] * 2)

        # mark action
        modifyArr(board, board_y - 1, board_x - 1, 0.5 * thrput_[i])
        modifyArr(board, board_y - 1, board_x    , 1.0 * thrput_[i])
        modifyArr(board, board_y - 1, board_x + 1, 0.5 * thrput_[i])

        modifyArr(board, board_y    , board_x - 1, 1.0 * thrput_[i])
        modifyArr(board, board_y    , board_x    , 1.2 * thrput_[i])
        modifyArr(board, board_y    , board_x + 1, 1.0 * thrput_[i])

        modifyArr(board, board_y + 1, board_x - 1, 0.5 * thrput_[i])
        modifyArr(board, board_y + 1, board_x    , 1.0 * thrput_[i])
        modifyArr(board, board_y + 1, board_x + 1, 0.5 * thrput_[i])

    # make input data based on current HAP location

    #### INPUT
    # board (center: x and y of UAV)
    center_x = int(q_current[0] * 2)
    center_y = int(q_current[1] * 2)

    input_board = board[center_y - 10 : center_y + 11, center_x - 10 : center_x + 11]

    # height of UAV
    height = q_current[2]

    # action (dif of x, y, and h of UAV)
    action = np.array(q_after) - np.array(q_current)

    print(' < action >')
    print(action)

    #### OUTPUT
    # compute output (reward) based on throughput change
    output = min(thrput_after) + 1.0 / np.std(thrput_after)

    # plot the board array using seaborn
    plt.clf()
    ax = sns.heatmap(input_board, annot=True)
    plt.savefig('input_board_' + str(l) + ',' + str(t) + '.png', bbox_inches='tight', dpi=100)

    # save training dataset
    # later

def throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU,
                   iterationCount, minThroughputList):

    # create list of devices (randomly place devices)
    deviceList = []

    for i in range(devices):
        xVal = random.random() * width
        yVal = random.random() * height
        device_i = [xVal, yVal]
        deviceList.append(device_i)

    # clustering

    # create the list of [cluster -> the number of devices] for each cluster and use it
    # for example: 3 devices in cluster 0, 7 devices in cluster 1, and 6 devices in cluster 2
    # then, it becomes [3, 7, 6]
    while True:
        (q, w, cluster_mem, markerColors) = algo.kMeansClustering(L, deviceList, width, height, H, N, False, True)
        numOfDevs = [cluster_mem.count(l) for l in range(L)]
        if min(numOfDevs) > 0: break

    # save device info
    print('< q >')
    print(np.array(q))

    print('\n< w >')
    for _ in w: print(_)

    print('\n< number of devices >')
    print(numOfDevs)

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
            devices = numOfDevs[j]
        
            for k in range(devices):
                for t in range(N):
                    alkl.append([i, k, j, t, 0])

    # sort alkl array
    alkl.sort(key=lambda x:x[3])
    alkl.sort(key=lambda x:x[2])
    alkl.sort(key=lambda x:x[1])
    alkl.sort(key=lambda x:x[0])

    # al : al[0] for each UAV l
    al = []
    for l in range(L): al.append(algo.transferEnergy(l, l))

    # speed of light
    c = 300000000

    minthroughputs  = []
    all_throughputs = []
    
    for l in range(L):

        # throughput results
        throughputs = []

        # the number of devices in cluster l
        devices = numOfDevs[l]
        
        for t in range(N):
            if printDetails == True:
                print('cluster ' + str(l) + ' / ' + str(L) + ', time ' + str(t) + ' / ' + str(N))

            # decide the device to communicate with
            #print(' ==== findDeviceToCommunicate ====')
            deviceToCommunicate = algo.findDeviceToCommunicate(q, w, l, t, N, s, b1, b2,
                                                               mu1, mu2, fc, c, alphaP, numOfDevs)
            #print(' =================================')

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
            thrputs = f.formula_11(q, w, l, k, alphaP, N, T, s, b1, b2, mu1, mu2, fc, c, L, alkl, PU, numOfDevs)
            thrput  = thrputs[-1]

            if printDetails == True:
                print('l=' + str(l) + ' k=' + str(k) + ' throughput=' + str(thrput))
            
            throughputs.append(thrputs)

        #### make training dataset ####
        for t in range(N-1):
            makeTrainDataset(w, l, directionList[l * N : (l + 1) * N], throughputs, t, q)

        # throughputs: shape (k, N) -> shape (N, k)
        throughputs       = np.array(throughputs)
        throughputs       = throughputs.T
        final_throughputs = throughputs[-1]

        # save throughputs
        if iterationCount < 5:
            throughputs_df = pd.DataFrame(throughputs)
            throughputs_df.to_csv('thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '.csv')

            final_throughputs_df = pd.DataFrame(final_throughputs)
            final_throughputs_df.to_csv('thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '_final.csv')

        # print average throughput result for each UAV
        print('\n\n ==== UAV ' + str(l) + ' ====')
        print('trajectory:')
        
        for t in range(N+1):
            if t < N:
                print('t:', t, 'x:', round(q[l * (N+1) + t][2], 4), 'y:', round(q[l * (N+1) + t][3], 4),
                      'h:', q[l * (N+1) + t][4], 'direction:', directionList[l * N + t])
            else:
                print('t:', t, 'x:', round(q[l * (N+1) + t][2], 4), 'y:', round(q[l * (N+1) + t][3], 4),
                      'h:', q[l * (N+1) + t][4])
                
        print('\nthroughputs for each device:')
        print(str(list(np.round_(final_throughputs, 6))))

        print('min throughput = ' + str(round(min(final_throughputs), 6)))
        minthroughputs.append(min(final_throughputs))
        
        print('===============\n\n')

        # save at all_throughputs
        all_throughputs += list(final_throughputs)

    # create min throughput information
    print('\n\nMIN THROUGHPUTS for each cluster l:')
    print(str(list(np.round_(minthroughputs, 6))))

    minThroughputList.append([iterationCount] + minthroughputs)

    # print all_throughputs
    all_throughputs = np.array(all_throughputs)

    print('\n\nall throughputs:')
    print(np.round_(all_throughputs, 6))

    all_throughputs = (all_throughputs - np.min(all_throughputs)) / (np.max(all_throughputs) - np.min(all_throughputs))

    print('\n\n(normalized) all throughputs:')
    print(np.round_(all_throughputs, 6))

    # save trajectory graph
    plt.clf()
    plt.suptitle('trajectory result at iter ' + str(iterationCount))
    plt.axis([-1, width+1, -1, height+1])

    # w = [[l, k, xkl, ykl, 0], ...]
    for i in range(len(w)):

        # device throughput = 0
        if all_throughputs[i] == 0:
            plt.scatter(w[i][2], w[i][3], s=25,
                        marker='o', c=markerColors[w[i][0]])

        # device throughput > 0
        else:
            plt.scatter(w[i][2], w[i][3],
                        s=int(pow(5.0 + all_throughputs[i] * 7.0, 2.0)),
                        marker='o', c=changeColor(markerColors[w[i][0]], 0.6))

    # q = [[l, t, xlt, ylt, hlt], ...]
    for l in range(L):
        for t in range(N+1):
            ind = l * (N+1) + t
           
            plt.scatter(q[ind][2], q[ind][3], s=25, marker='x', c=markerColors[l])

            if t < N:
                x = [q[ind][2], q[ind+1][2]]
                y = [q[ind][3], q[ind+1][3]]
                plt.plot(x, y, linewidth=0.75, c=markerColors[l])

    # save the figure
    plt.savefig('trajectory_iter' + ('%04d' % iterationCount))

if __name__ == '__main__':

    # list of min throughputs
    minThroughputList = []

    # to bugfix:
    # [solved] 1. at the end of device list w = [[l, k, xkl, ykl, 0], ...]
    # [solved] 2. no device (l, k) in device list w = [[l, k, xkl, ykl, 0], ...] (at formula_02 and getInferencekl)
    # [solved] 3. no device (l, k) in device list w = [[l, k, xkl, ykl, 0], ...] (at find_alkl and findDeviceToCommunicate)

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
                                'M':'int', 'L':'int', 'devices':'int', 'T':'float', 'N':'int', 'H':'float',
                                'iters':'int'})

    fc = paperArgs['fc']
    ng = paperArgs['ng']
    B = paperArgs['B']
    o2 = paperArgs['o2']
    b1 = paperArgs['b1']
    b2 = paperArgs['b2']
    alphaP = paperArgs['alphaP']
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
    iters = paperArgs['iters']

    # (from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1)
    # SN = (1 − alphaP)*T/N denotes the length of each subslot
    # for uplink, alphaP stands for the proportion of downlink WPT in a period.

    # Specifically, the 0-th time slot is assigned to the downlink
    # WPT and the n-th time slot, n in N = {1,2,...,N} is allocated to the uplink WIT.
    alphaL = 1/(N+1)

    # write file for configuration info
    configFile = open('config_info.txt', 'w')
    
    configContent = 'fc=' + str(fc) + '\n'
    configContent += 'ng=' + str(ng) + '\n'
    configContent += 'B=' + str(B) + '\n'
    configContent += 'o2=' + str(o2) + '\n'
    configContent += 'b1=' + str(b1) + '\n'
    configContent += 'b2=' + str(b2) + '\n'
    configContent += 'alphaP=' + str(alphaP) + '\n'
    #configContent += 'alphaL=' + str(alphaL) + '\n'
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
    configContent += 'H=' + str(H) + '\n'
    configContent += 'iters=' + str(iters)

    configFile.write(configContent)
    configFile.close()

    # run throughput test
    #throughputTest(M, T, N, L, devices, width, height, H,
    #               ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU)
    for iterationCount in range(iters):
        print('ITER COUNT ', iterationCount)
        
        throughputTest(M, T, N, L, devices, width, height, H,
                       ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                       iterationCount, minThroughputList)

    # save min throughput list as *.csv file
    minThroughputList = pd.DataFrame(np.array(minThroughputList))
    minThroughputList.to_csv('minThroughputList_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                             '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.csv')

    # save min throughput list as *.txt file
    arr = np.array(minThroughputList)[:, 1:]
    note = 'mean: ' + str(np.mean(arr)) + ', std: ' + str(np.std(arr)) + ', nonzero: ' + str(np.count_nonzero(arr))

    noteFile = open('minThroughputList_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                             '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.txt', 'w')
    
    noteFile.write(note)
    noteFile.close()
