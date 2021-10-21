import sys

import helper as h_
import formula as f
import deepQ as dq
import algorithms as algo

import math
import random
import copy
import numpy as np
from shapely.geometry import LineString

import tensorflow as tf
import time

timeCheck = h_.loadSettings({'timeCheck':'logical'})['timeCheck']

# time consumption per time slot : 38 seconds -> 3.0 seconds (about 1/13)

# PAPER : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1

# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is THE POSITION for x, y and h at time t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...

# that is, in the form of UAV[uav_No][x/y/h][time]

# check if UAV i files beyond the border
# (THERE IS NO CASE that location of UAV at time t-1 and t is both within the border,
#  but the UAV files beyond the border between time t-1 and t.) 
def beyondBorder(UAVi, t, width, height):
    
    xi = UAVi[0] # x axis for UAVi
    yi = UAVi[1] # y axis for UAVi
    hi = UAVi[2] # height for UAVi

    #print('====', t, '====')
    #for i in range(len(xi)):
    #    print('time', i, xi[i], yi[i], hi[i])

    if xi[t-1] < 0 or xi[t-1] > width:
        #print('beyond border:', xi, yi, hi, t)
        return True
    
    elif xi[t] < 0 or xi[t] > width:
        #print('beyond border:', xi, yi, hi, t)
        return True
    
    elif yi[t-1] < 0 or yi[t-1] > height:
        #print('beyond border:', xi, yi, hi, t)
        return True
    
    elif yi[t] < 0 or yi[t] > height:
        #print('beyond border:', xi, yi, hi, t)
        return True

    return False

# check if UAV i's trajectory and UAV j's trajectory is crossed
def IsTrajectoryCrossed(UAVi, UAVj, t):
    
    # trajectory: location of UAV i between time slot t-1 ~ t
    #             location of UAV j between time slot t-1 ~ t

    # get position of UAVi and UAVj at time t-1 and t
    xi_before = UAVi[0][t-1]
    yi_before = UAVi[1][t-1]
    hi_before = UAVi[2][t-1]
    xi_after = UAVi[0][t]
    yi_after = UAVi[1][t]
    hi_after = UAVi[2][t]

    xj_before = UAVj[0][t-1]
    yj_before = UAVj[1][t-1]
    hj_before = UAVj[2][t-1]
    xj_after = UAVj[0][t]
    yj_after = UAVj[1][t]
    hj_after = UAVj[2][t]

    # decide there is crossing of trajectory
    UAVi_before = (xi_before, yi_before, hi_before) # location of UAVi at time t-1
    UAVi_after = (xi_after, yi_after, hi_after) # location of UAVi at time t
    
    UAVj_before = (xj_before, yj_before, hj_before) # location of UAVj at time t-1
    UAVj_after = (xj_after, yj_after, hj_after) # location of UAVj at time t

    # check if the segments intersect
    UAVi_line = LineString([UAVi_before, UAVi_after])
    UAVj_line = LineString([UAVj_before, UAVj_after])

    return UAVi_line.intersects(UAVj_line)

# for compatibility between UAVi[0...2][t] and (x, y or h)[n][l] (or q[n][l])
# UAVs    : list of UAV -> [UAV0, UAV1, ...], shape of (# UAVs, 3(for x, y and h), # time slots)
# element : 'x' for x, 'y' for y, or 'h' for h
# T       : number of time slots
def getXYH_ofUAV(UAVs, element, T):

    # decide what to return (x -> UAVs[UAV][0][t], y -> UAVs[UAV][1][t], h -> UAVs[UAV][2][t])
    if element == 'x': xyh = 0
    elif element == 'y': xyh = 1
    elif element == 'h': xyh = 2

    # the format of result : q[n][l] where n indicates time slot and l indicates each UAV
    result = []

    # create and return result array
    for t in range(len(T)): # for each time slot ('t' in UAVs and 'n' in q)
        forTimeSlotT = [] # q[n]
        for UAV in range(len(UAVs)): # for each UAV ('UAV' in UAVs and 'l' in q)
            forTimeSlotT.append(UAVs[UAV][xyh][t]) # q[n][l]
        result.append(forTimeSlotT)

    return result

# check if minimum throughput of all devices in a cluster == 0
# ASSUMPTION: 'a cluster' means the cluster with index l
def isMinThroughputOfAllDevicesInCluster0(cluster, L, T, l, k, a, B, n, PU, g, o2):

    # minimum throughput == 0 <=> there are some devices with throughput == 0
    # clusters[l] = cluster l, and k[l] = a device in K[l] (cluster l)
    for k in range(len(cluster)):
        thrput = f.R_kl(L, T, l, k, a, B, n, PU, g, o2)
        if thrput == 0: return True # return True if the throughput is 0

    return False

# check if minimum throughput of all devices does not increase
# ASSUMPTION: between time n-1(= t-1), time n(= t)
def isMinThroughputOfAllDevicesDoesNotInc(cluster, L, T, l, k, a, B, n, PU, g, o2):

    # clusters[l] = cluster l (total L clusters), and k[l] = a device in K[l] (cluster l)
    for l in range(L):

        # get throughput for (cluster l, device k)
        for k in range(len(cluster)):
            thrputBefore = f.R_kl(L, T, l, k, a, B, n-1, PU, g, o2) # throughput at time n-1 = t-1
            thrputAfter = f.R_kl(L, T, l, k, a, B, n, PU, g, o2) # throughput at time n = t

        # initialize minThroughputBefore and minThroughputAfter as that of (cluster l=0, device k=0)
        if l == 0 and k == 0:
            minThroughputBefore = thrputBefore
            minThroughputAfter = thrputAfter
            
        # else, update them
        else:
            if thrputBefore < minThroughputBefore: minThroughputBefore = thrputBefore
            if thrputAfter < minThroughputAfter: minThroughputAfter = thrputAfter

    # compare minThroughputBefore and minThroughputAfter, and return
    if minThroughputBefore < minThroughputAfter: return True
    else: return False

# get cluster where device [X0, Y0] is located in
## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]

# device   : in the form of [X0, Y0]
# returnNo : return the No.(index) of the cluster (if False, return the cluster itself)
def getClusterOfDevice(device, clusters):
    for i in range(clusters):
        if device in clusters[i]:
            if returnNo == True: return i # return the number
            else: return cluster[i] # return the cluster

# get x, y and h from UAVs

# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is THE POSITION for x, y and h at time t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...
# -> x[time][UAV], y[time][UAV] and h[time][UAV]
def xyh(UAVs, xyh_):

    # xyh_: 0(for x), 1(for y) or 2(for h)
    # for example, # UAVs[l] = [x0, y0, h0], UAVs[l][0] = x0 = [x00, x01, ..., x0t]
    result = []

    for t in range(len(UAVs[0][0])): # for each time slot (t)
        temp = []

        for l in range(len(UAVs)): # for each UAV (l)
            temp.append(UAVs[l][xyh_][t])
            
        result.append(temp)

    return result

# get x
def x(UAVs):
    return xyh(UAVs, 0)

# get y
def y(UAVs):
    return xyh(UAVs, 1)

# get h
def h(UAVs):
    return xyh(UAVs, 2)

# update direct reward list for UAV i
# cluters           : in the form of [list([[x of device 0, y of device 0], ...]), ...]
# action            : action [x, y, z]
# a                 : array {a[n][l][k_l]}
# alpha             : value of alpha
# directReward_list : direct reward list to update
# updatingQtable    : update Q table?
def updateDRlist(UAVs, value, i, deviceList, b1, b2, S_, u1, u2, fc, n, action, a,
                 Q, QTable, s_i, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                 trainedModel, optimizer, updatingQtable, printTimeDif):

    if timeCheck == True: h_.printTime('updateDRlist', 'IN')
    
    # for each device k
    for k in range(len(clusters[i])):

        # get PLoS and PNLoS
        x_UAV = x(UAVs)[n][i]
        y_UAV = y(UAVs)[n][i]
        h_UAV = h(UAVs)[n][i]
        
        PLoS_i = f.getPLoS(False, i, k, clusters, x_UAV, y_UAV, h_UAV, b1, b2, S_)
        PNLoS_i = 1.0 - PLoS_i
            
        # update Q value
        currentTime = getTimeDif(None, '(updateDRlist) before g_nlkl', printTimeDif)
        g_i = f.g_nlkl(PLoS_i, u1, PNLoS_i, u2, fc, i, k, clusters, x_UAV, y_UAV, h_UAV, alpha)

        currentTime = getTimeDif(currentTime, '(updateDRlist) after g_nlkl', printTimeDif)

        if updatingQtable == True:
            dq.updateQvalue(Q, QTable, s_i, action, a, value, alphaL, r_, n, UAVs, i, useDL, clusters, B, PU, g, o2, L,
                            trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha)

            currentTime = getTimeDif(currentTime, '(updateDRlist) after updateQvalue', printTimeDif)

        if k == 0: directReward_list[i] += value

    if timeCheck == True: h_.printTime('updateDRlist', 'OUT')

# print time difference
def getTimeDif(T, msg, printTimeDif):

    if printTimeDif == False: return

    # print time difference
    newT = time.time()
    
    try:
        print('[' + msg + '] time dif : ' + str(round(newT - T, 6)) + ' seconds')
    except:
        print('[' + msg + ']')
    
    return newT

# print warning
def getWarning(printWarning, text, value):

    if printWarning == True: print('[' + str(value) + '] ' + text)

# to do:
# make state transition algorithm (the best state)
# -> find and use communication count 'a' during translation

# ALGORITHM 1
# 3D trajectory design and time resource allocation solution based on DQL
# M                  : number of episodes
# T                  : number of time slots
# L                  : number of clusters = number of UAVs
# H                  : height of UAVs
# devices            : number of devices
# width, height      : width and height of the board
# fc, B, o2, b1, b2, : parameters (as Table 1)
# alpha, u1, u2,
# alphaL, r_, S_
# deviceName         : cpu:0 or gpu:0 for example
# QTable_rate        : rate of used elements from Q Table (most recently)
def algorithm1(M, T, L, devices, width, height,
               H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_, S_,
               deviceName, QTable_rate, iteration):

    # load settings
    printTimeDif = h_.loadSettings({'printTimeDif':'logical'})['printTimeDif']
    printWarning = h_.loadSettings({'warning':'logical'})['warning']
    
    ### INIT ###
    # Q Table: [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    QTable = [] # Q table

    # info about all devices: [dev0, dev1, ...] = [[X0, Y0], [X1, Y1]]
    deviceList = []

    # init UAV's location and IoT devices' location
    
    # **** UAVs: in the form of list : [UAV0, UAV1, ...] ****
    # each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
    # each element of x0, y0 and h0 is THE POSITION for x, y and h at time t
    # where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
    #       y0 = [y00, y01, ..., y0t], ...
    #       h0 = [h00, h01, ..., h0t], ...

    # that is, in the form of UAV[uav_No][x/y/h][time]

    # **** clusters: in the form of list : [c0_deviceList, c1_deviceList, ...] ****
    # cK_deviceList: device list of cluster k,
    #                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]

    # randomly place devices
    for i in range(devices):
        xVal = random.random() * width
        yVal = random.random() * height
        device_i = [xVal, yVal]
        deviceList.append(device_i)

    # cluster UAVs using K-means clustering
    # clusters (number of clusters = L, number of total devices = devices)
    while True:
        (UAVs, clusters, clusterNOs) = algo.kMeansClustering(L, deviceList, width, height, H, T, False, True)

        # check if all cluster has at least 1 devices
        # the value of len(list(set(clusterNOs))) is L+1 when all cluster has at least 1 devices
        if len(list(set(clusterNOs))) - 1 == L:
            break

    # no need to init target network and online network now

    # create action space
    actionSpace = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                actionSpace.append([i, j, k])

    # zero_27 array (initial action array)
    zero_27 = []
    for i in range(27): zero_27.append(0)

    # init ac and R where
    # a[n][l][k_l] (a[n][l][k])    : the number of times that each device communicates with UAV l
    # g[n][l][k_l] (g[n][l][k][l]) : the channel's power gain between UAV l and device k_l
    # R[n][k_l]    (R[n][l][k])    : the average throughput of devices (for each device k),
    #                                in l-th cluster (1d array, for the devices in l-th cluster)
    # PU[n][k_l]   (PU[n][l][k])   : peak power for IoT devices' uplink transmit (at most -20dBm -> assumption: -20dBm)
    a = []
    g = []
    PU = []

    for t in range(T + 1): # n = 0,1,...,T (# of time slots)

        # not n = 0,1,...,T-1 to prevent bug
        temp_a = []
        temp_g = []
        temp_PU = []
        
        for l in range(L): # l = 0,1,...,L-1 (# of clusters = # of UAVs)
            temp_a_ = []
            temp_g_ = []
            temp_PU_ = []
            
            K = len(clusters[l]) # number of devices for UAV l

            for k in range(K): # k = 0,...,K-1 (# of devices for each UAV l)
                temp_a_.append(0)
                temp_PU_.append(-20)

                temp_g__ = []
                for ll in range(L): # ll = 0,1,...,L-1 (# of clusters = # of UAVs)
                    temp_g__.append(0)
                temp_g_.append(temp_g__)
                
            temp_a.append(temp_a_)
            temp_g.append(temp_g_)
            temp_PU.append(temp_PU_)
                
        a.append(temp_a)
        g.append(temp_g)
        PU.append(temp_PU)
  
    # init Q Table
    # Q Table = [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    Q = []

    ### DATA COLLECT ###
    # REPLAY BUFFER: later

    currentTime = getTimeDif(None, 'init', printTimeDif)
    
    for episode in range(M):
        print('\nepisode ' + str(episode) + ' / ' + str(M) + ' / time=' + str(time.time()))

        # choose action with e-greedy while e increases
        e_ = episode / M # example

        # use deep learning if length of Q table >= 100
        # load trained model for episode = 1, 2, ...
        if len(QTable) > 0 and episode > 0:
            useDL = True
            trainedModel = tf.keras.models.load_model('model')
            optimizer = tf.keras.optimizers.Adam(0.001)
            
        else:
            useDL = False
            trainedModel = None
            optimizer = None

        # initialize a
        for i in range(len(a)):
            for j in range(len(a[i])):
                for k in range(len(a[i][j])):
                    a[i][j][k] = 0

        # save {a[n][l][k_l]}, {R[n][k_l]} of state = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
        # shape: (T+1, L, 2)
        savedStates = []
        
        for i in range(T+1):
            savedStates_ = []
            
            for j in range(L):
                savedStates_.append([None, None])

            savedStates.append(savedStates_)

        # rows to be created: (for training data) when QTable_rate = 1.0 :
        # [T = time slots (second)] * [devices = number of devices] * 3     
        for t in range(1, T): # each time slot t
            print('time slot ' + str(t - 1) + ' / ' + str(T))
            print('[00]', time.time())

            currentTime = getTimeDif(currentTime, 'start of time slot', printTimeDif)

            # initialize direct reward
            directReward_list = [] # direct reward for action (for each UAV)
            for i in range(L): directReward_list.append(0) # initialize as 0
            
            oldS_list = [] # old states (for each UAV)
            action_list = [] # actions (for each UAV)
            newS_list = [] # new states (for each UAV)
            
            for i in range(L): # for each UAV = each cluster

                currentTime = getTimeDif(currentTime, '0', printTimeDif)
                
                # get UAV i's next location
                s_i = dq.getS(UAVs[i], t, i, a, L, B, PU, g, o2)

                # choose action with e-greedy while e increases
                action = dq.getActionWithE(Q, s_i, e_)

                #print('current UAV location:', [UAVs[i][0][t], UAVs[i][1][t], UAVs[i][2][t]])
                nextLocation = dq.getNextLocation(s_i, action, t, UAVs, l, a, L, B, PU, g, o2)

                # update UAV location for the next time t+1
                nextX = nextLocation[0]
                nextY = nextLocation[1]
                nextH = nextLocation[2]

                for tt in range(t, T+1):
                    #if tt == t: print(UAVs[i][0][t], UAVs[i][1][t], UAVs[i][2][t])
                    UAVs[i][0][tt] = nextX
                    UAVs[i][1][tt] = nextY
                    UAVs[i][2][tt] = nextH
                    #if tt == t: print(UAVs[i][0][t], UAVs[i][1][t], UAVs[i][2][t])
                    
                # if UAV i files beyond the border
                #print('nextLocation:', nextLocation, [UAVs[i][0][t], UAVs[i][1][t], UAVs[i][2][t]])
                
                if beyondBorder(UAVs[i], t, width, height) == True:
                    getWarning(printWarning, 'beyond border', i)

                    # UAV i stays at the border
                    for tt in range(t, T+1):
                        if UAVs[i][0][tt] < 0.0: UAVs[i][0][tt] = 0.0 # x value < 0
                        elif UAVs[i][0][tt] > width: UAVs[i][0][tt] = width # x value > width
                        
                        if UAVs[i][1][tt] < 0.0: UAVs[i][1][tt] = 0.0 # y value < 0
                        elif UAVs[i][1][tt] > height: UAVs[i][1][tt] = height # y value > height

                        if UAVs[i][2][tt] < 0.0: UAVs[i][2][tt] = 0.0 # h value < 0
                    
                    # UAV i gets a penalty of -1
                    currentTime = getTimeDif(currentTime, '0 before updateDRlist', printTimeDif)
                    updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, QTable, s_i, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                                 trainedModel, optimizer, False, printTimeDif)
                    currentTime = getTimeDif(currentTime, '0 after updateDRlist', printTimeDif)

            for i in range(L): # each UAV i
                currentTime = getTimeDif(currentTime, '1', printTimeDif)
                
                for j in range(i): # each UAV j

                    # UAV i and j's trajectory exists cross
                    if IsTrajectoryCrossed(UAVs[i], UAVs[j], t) == True:
                        getWarning(printWarning, 'trajectory crossed', i)
                        getWarning(printWarning, 'trajectory crossed', j)
                        
                        # UAV i and UAV j stays at the previous position
                        UAVs[i][0][t] = UAVs[i][0][t-1]
                        UAVs[i][1][t] = UAVs[i][1][t-1]
                        UAVs[j][0][t] = UAVs[j][0][t-1]
                        UAVs[j][1][t] = UAVs[j][1][t-1]

                        # UAV i and UAV j get a penalty of -1
                        s_i = dq.getS(UAVs[i], t, i, a, L, B, PU, g, o2)

                        # (for i) choose action with e-greedy while e increases
                        action_i = dq.getActionWithE(Q, s_i, e_)

                        currentTime = getTimeDif(currentTime, '1-0 before updateDRlist', printTimeDif)
                        updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action_i, a,
                                     Q, QTable, s_i, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                                     trainedModel, optimizer, False, printTimeDif)

                        s_j = dq.getS(UAVs[j], t, j, a, L, B, PU, g, o2)

                        # (for j) choose action with e-greedy while e increases
                        action_j = dq.getActionWithE(Q, s_j, e_)

                        currentTime = getTimeDif(currentTime, '1-1 before updateDRlist', printTimeDif)
                        updateDRlist(UAVs, -1, j, deviceList, b1, b2, S_, u1, u2, fc, t, action_j, a,
                                     Q, QTable, s_j, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                                     trainedModel, optimizer, False, printTimeDif)
                        currentTime = getTimeDif(currentTime, '1-1 after updateDRlist', printTimeDif)

                # get throughput (before) (time = n) (n = t, l = i, k = t)
                # error if (time slot value) > (devices)
                try:
                    beforeThroughput = f.R_nkl(L, B, t, i, min(t, len(PU[t][i])-1), PU, g, o2)
                except:
                    print('cannot find throughput (before) because there are no devices in cluster ' + str(i))
                    continue

                #### HERE ####
                
                # get and move to next state (update q, a and R)
                # s       : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
                # q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
                # action  : action ([-1, -1, -1] to [1, 1, 1])

                # "Don't be confused between s = [q, a, R] and [state, action, newState]"

                # compute initial state or get old state from savedStates
                s = dq.getS(UAVs[i], t, i, a, L, B, PU, g, o2)
                action_s = dq.getActionWithE(Q, s, e_)
                
                if t > 1:
                    s[1] = savedStates[t-1][i][0]
                    s[2] = savedStates[t-1][i][1]

                # save current state before getting next state
                oldS = copy.deepcopy(s)
                
                # get next state
                (nextState, _) = dq.getNextState(s, action_s, t, UAVs, i, a, clusters, B, PU, g, o2, L,
                                                 b1, b2, S_, u1, u2, fc, alpha)
                
                q_next = copy.deepcopy(nextState[0])
                a_next = copy.deepcopy(nextState[1])
                R_next = copy.deepcopy(nextState[2][i])
                
                savedStates[t][i] = [a_next, R_next]
                s = [q_next, a_next, R_next] # update current state
                
                # append to oldS_list and newS_list
                oldS_list.append(oldS)
                newS_list.append(s)
                action_list.append(action)
                
                # get throughput (after) (time = t+1) (n = t+1, l = i, k = t)
                # error if (time slot value) > (devices)
                try:
                    afterThroughput = f.R_nkl(L, B, t+1, i, min(t, len(PU[t+1][i])-1), PU, g, o2)
                except:
                    print('cannot find throughput (after) because there are no devices in cluster ' + str(i))
                    continue

                # device t's throughput does not increase
                # assumption: device t is the device communicated with when the time slot is t
                # [temporary] assume that the index of the device is t
                if afterThroughput <= beforeThroughput:
                    getWarning(printWarning, 'throughput did not increase', i)

                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAVs[i], t, i, a, L, B, PU, g, o2)
                    action_si = dq.getActionWithE(Q, s_i, e_)

                    currentTime = getTimeDif(currentTime, '2 before updateDRlist', printTimeDif)
                    updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action_si, a,
                                 Q, QTable, s_i, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                                 trainedModel, optimizer, False, printTimeDif)
                    currentTime = getTimeDif(currentTime, '2 after updateDRlist', printTimeDif)

            # if time slot is T
            if t == T:

                # for each UAV = cluster,
                for i in range(L):
                    
                    # if minimum throughput of all devices in a cluster == 0
                    if isMinThroughputOfAllDevicesInCluster0(clusters[i], L, T, l, k, a, B, t, PU, g, o2) == True:
                        getWarning(printWarning, 'min throughput == 0', i)

                        # The UAV get a penalty of -2
                        directReward_list[i] += (-2)

                # if minimum throughput of all devices does not increase
                if isMinThroughputOfAllDevicesDoesNotInc(clusters[i], L, T, l, k, a, B, t, PU, g, o2) == True:
                    getWarning(printWarning, 'min throughput does not increase', i)

                    # All UAVs get a penalty of -1
                    index = 0
                    
                    for UAV in UAVs:
                        s_UAV = dq.getS(UAV, t, index, a, L, B, PU, g, o2)
                        action_sUAV = dq.getActionWithE(Q, s_UAV, e_)

                        currentTime = getTimeDif(currentTime, '3 before updateDRlist', printTimeDif)
                        updateDRlist(UAVs, -1, index, deviceList, b1, b2, S_, u1, u2, fc, t, action_sUAV, a,
                                     Q, QTable, s_UAV, alpha, alphaL, r_, useDL, clusters, B, PU, o2, L, directReward_list, g,
                                     trainedModel, optimizer, False, printTimeDif)
                        currentTime = getTimeDif(currentTime, '3 after updateDRlist', printTimeDif)
                        
                        index += 1

            # update Q Table and store (s,a,r,s') into replay buffer (later)
            for i in range(L): # each UAV i
                currentTime = getTimeDif(currentTime, '4', printTimeDif)

                # for each device k,
                # reward = alphaL*(Rwd(s, a) + r_*maxQ)
                #        = alphaL*(Rwd(s, a) + r_*max(a')Q(s', a'))
                # where maxQ = max(a')Q(s', a')
                maxQs = []
                
                # for each device k
                for k in range(len(clusters[i])):

                    x_UAV = x(UAVs)[t][i]
                    y_UAV = y(UAVs)[t][i]
                    h_UAV = h(UAVs)[t][i]
        
                    PLoS_i = f.getPLoS(False, i, k, clusters, x_UAV, y_UAV, h_UAV, b1, b2, S_)
                    PNLoS_i = 1.0 - PLoS_i

                    # update g (g[n][l][k_l]) : the channel's power gain between UAV l and device k_l
                    g_i = f.g_nlkl(PLoS_i, u1, PNLoS_i, u2, fc, i, k, clusters, x_UAV, y_UAV, h_UAV, alpha)
                    g[t][i][k][i] = g_i

                currentTime = getTimeDif(currentTime, '4 before getMaxQ', printTimeDif)

                action_rewards = copy.deepcopy(zero_27)

                # compute reward for each action (ALL 27 POSSIBLE ACTIONS)
                for j in range(27):
                    
                    # first iteration, trainedModel does not exist
                    if episode == 0:
                        (maxQ, _) = dq.getMaxQ(oldS_list[i], dq.getAction(j), t, UAVs, i, a, actionSpace, clusters, B, PU, g, o2, L,
                                               useDL, None, optimizer, b1, b2, S_, u1, u2, fc, alpha)
                    
                    # trainedModel exists -> execute
                    else:
                        (maxQ, _) = dq.getMaxQ(oldS_list[i], dq.getAction(j), t, UAVs, i, a, actionSpace, clusters, B, PU, g, o2, L,
                                               useDL, trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha)
                        
                    currentTime = getTimeDif(currentTime, '4 after getMaxQ', printTimeDif)
                        
                    reward = alphaL * (directReward_list[i] + r_ * maxQ)
                    maxQs.append(maxQ)
                    
                    # append to Q Table: [[[s0], [Q00, Q01, ...]], [[s1], [Q10, Q11, ...]], ...]
                    # where s = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
                    # and   Q = reward
                    # from oldS_list, action_list and reward
                    action_rewards[j] = reward

                QTable.append([oldS_list[i], action_rewards, i])

        ### TRAIN and VALIDATION ###
        if episode == M or (episode < 100 and episode % 10 == 0) or episode % 50 == 0:
            QTable_use = len(QTable) * QTable_rate
            dq.deepLearningQ_training(QTable[len(QTable) - int(QTable_use):], deviceName, 10, False, iteration, M, episode, clusters)

    ### TRAIN and VALIDATION (FINAL) ###
    QTable_use = len(QTable) * QTable_rate
    dq.deepLearningQ_training(QTable[len(QTable) - int(QTable_use):], deviceName, 10, False, iteration, M, M, clusters)

    ### TEST ###
    # later

def convertToNumeric(value):
    try:
        return float(value)
    except:
        return float(int(value))

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

    # run for 20 iterations
    iters = 20
    
    for i in range(iters):
        algorithm1(M, T, L, devices, width, height,
                   H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_, S_,
                   deviceName, QTable_rate, i)
