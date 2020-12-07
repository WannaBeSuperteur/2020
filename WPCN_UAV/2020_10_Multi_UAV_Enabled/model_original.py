import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main

import formula as f
import deepQ as dq
import algorithms as algo

import math
import random
import copy
import numpy as np
from shapely.geometry import LineString

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

    if xi[t-1] < 0 or xi[t-1] > width: return True
    elif xi[t] < 0 or xi[t] > width: return True
    elif yi[t-1] < 0 or yi[t-1] > height: return True
    elif yi[t] < 0 or yi[t] > height: return True

    return True

# check if UAV i's trajectory and UAV j's trajectory is crossed
def IsTrajectoryCrossed(UAVi, UAVj, t):
    
    # trajectory: location of UAV i between time slot t-1 ~ t
    #             location of UAV j between time slot t-1 ~ t

    # get position of UAVi and UAVj at time t-1 and t
    xi_before = UAVi[0][t-1]
    yi_before = UAVi[1][t-1]
    xi_after = UAVi[0][t]
    yi_after = UAVi[1][t]

    xj_before = UAVj[0][t-1]
    yj_before = UAVj[1][t-1]
    xj_after = UAVj[0][t]
    yj_after = UAVj[1][t]

    # decide there is crossing of trajectory
    UAVi_before = (xi_before, yi_before) # location of UAVi at time t-1
    UAVi_after = (xi_after, yi_after) # location of UAVi at time t
    UAVj_before = (xj_before, yj_before) # location of UAVj at time t-1
    UAVj_after = (xj_after, yj_after) # location of UAVj at time t

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
def isMinThroughputOfAllDevicesInCluster0(clusters, T, N, l, k, B, n, PU, g, I_, o2):

    # minimum throughput == 0 <=> there are some devices with throughput == 0
    # clusters[l] = cluster l, and k[l] = a device in K[l] (cluster l)
    for k in range(len(clusters[l])):
        thrput = R_kl(T, N, l, k, B, n, PU, g, I_, o2)
        if thrput == 0: return True # return True if the throughput is 0

    return False

# check if minimum throughput of all devices does not increase
# ASSUMPTION: between time n-1(= t-1), time n(= t)
def isMinThroughputOfAllDevicesDoesNotInc(T, N, l, k, B, n, PU, g, I_, o2):

    # clusters[l] = cluster l (total L clusters), and k[l] = a device in K[l] (cluster l)
    for l in range(L):

        # get throughput for (cluster l, device k)
        for k in range(len(clusters[l])):
            thrputBefore = R_kl(T, N, l, k, B, n-1, PU, g, I_, o2) # throughput at time n-1 = t-1
            thrputAfter = R_kl(T, N, l, k, B, n, PU, g, I_, o2) # throughput at time n = t

        # initialize minThroughputBefore and minThroughputAfter as that of (cluster l=0, device k=0)
        if l == 0 and k == 0:
            minThroughputBefore = thrputBefore
            minThroughputAfter = thrputAfter
            
        # else, update them
        else:
            if thrputBefore < minThroughputBefore: minThroughputBefore = thrputBefore
            if thrputAfter > minThroughputAfter: minThroughputAfter = thrputAfter

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
# alphaL, r_
def algorithm1(M, T, L, devices, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_):

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
    (UAVs, clusters) = algo.kMeansClustering(L, deviceList, width, height, H, T, False)

    # no need to init target network and online network now

    # init ac and R where
    # ac[n][l][k_l] (ac[n][l]) : the number of times that each device communicates with UAV l (1d array, for all devices)
    # R[n][k_l]     (R[n])     : the average throughput of devices (for each device k),
    #                            in l-th cluster (1d array, for the devices in l-th cluster)
    ac = []
    R = []
    
    for n in range(T): # n = 0,1,...,T-1 (# of time slots)
        temp_ac = []
        for l in range(L): # l = 0,1,...,L-1 (# of clusters = # of UAVs)
            temp_ac.append(0)
        ac.append(temp_ac)
        R.append(0)

    # init Q Table
    # Q Table = [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    Q = []

    ### TRAIN ###
    replayBuffer = [] # REPLAY BUFFER
    
    for episode in range(M):
        for t in range(T): # each time slot t

            directReward_list = [] # direct reward for action (for each UAV)
            for i in range(L): directReward_list.append(0) # initialize as 0
            
            oldS_list = [] # old states (for each UAV)
            action_list = [] # actions (for each UAV)
            newS_list = [] # new states (for each UAV)
            
            for i in range(L): # for each UAV = each cluster
                
                # get UAV i's next location
                s_i = dq.getS(UAVs[i], t, i, ac, R)

                # choose action with e-greedy while e increases
                e_ = episode / M # example
                a = dq.getActionWithE(Q, s_i, e_)
                
                nextLocation = dq.getNextLocation(s_i, a, t)

                # if UAV i files beyond the border
                if beyondBorder(UAVs[i], t, width, height) == True:
                    
                    # UAV i stays at the border
                    if UAVs[i][0][t] < 0: UAVs[i][0][t] = 0 # x value < 0
                    elif UAVs[i][0][t] > width: UAVs[i][0][t] = width # x value > width
                    
                    if UAVs[i][1][t] < 0: UAVs[i][1][t] = 0 # y value < 0
                    elif UAVs[i][1][t] > height: UAVs[i][1][t] = height # y value > height
                    
                    # UAV i gets a penalty of -1
                    dq.updateQvalue(Q, s_i, a, -1, alpha, r_, t, i, k, R, useDL)
                    directReward_list[i] += (-1)

            for i in range(L): # each UAV i
                
                for j in range(i): # each UAV j

                    # UAV i and j's trajectory exists cross
                    if IsTrajectoryCrossed(UAVs[i], UAVs[j], t) == True:

                        # UAV i and UAV j stays at the previous position
                        UAVs[i][0][t] = UAVs[i][0][t-1]
                        UAVs[i][1][t] = UAVs[i][1][t-1]
                        UAVs[j][0][t] = UAVs[j][0][t-1]
                        UAVs[j][1][t] = UAVs[j][1][t-1]

                        # UAV i and UAV j get a penalty of -1
                        s_i = dq.getS(UAVs[i], t, i, ac, k, R)
                        dq.updateQvalue(Q, s_i, a, -1, alpha, r_, t, i, k, R, useDL)
                        directReward_list[i] += (-1)

                        s_j = dq.getS(UAVs[j], t, j, ac, k, R)
                        dq.updateQvalue(Q, s_j, a, -1, alpha, r_, t, j, k, R, useDL)
                        directReward_list[j] += (-1)

                # get throughput (before) (time = n)
                beforeThroughput = f.R_nkl(B, k, i, t, PU, g, I_, o2)
                
                # get and move to next state (update q, a and R)
                # s       : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
                # q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
                # a       : action ([-1, -1, -1] to [1, 1, 1])
                
                oldS = copy.deepcopy(s) # save old state
                nextState = dq.getNextState(s, a, t, i, k, R, clusters, B, PU, g, l_, o2)
                q = copy.deepcopy(nextState[0])
                a = copy.deepcopy(nextState[1])
                R = copy.deepcopy(nextState[2])
                s = [q, a, R] # update current state

                # append to oldS_list and newS_list
                oldS_list.append(oldS)
                newS_list.append(s)
                action_list.append(a)

                # get throughput (after) (time = n+1)
                afterThroughput = f.R_nkl(B, k, i, n+1, PU, g, I_, o2)

                # device t's throughput does not increase
                if afterThroughput <= beforeThroughput:
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAVs[i], t, i, ac, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, r_, t, i, k, R, useDL)
                    directReward_list[i] += (-1)

            # if time slot is T
            if t == T:

                # for each UAV = cluster,
                for i in range(L):
                    
                    # if minimum throughput of all devices in a cluster == 0
                    if isMinThroughputOfAllDevicesInCluster0(clusters[i], t) == True:

                        # The UAV get a penalty of -2
                        directReward_list[i] += (-2)

                # if minimum throughput of all devices does not increase
                if isMinThroughputOfAllDevicesDoesNotInc(t) == True:

                    # All UAVs get a penalty of -1
                    index = 0
                    
                    for UAV in UAVs:
                        s_UAV = dq.getS(UAV, t, index, ac, k, R)
                        dq.updateQvalue(Q, s_UAV, a, -1, alpha, r_, t, index, k, R, useDL)
                        directReward_list[UAV] += (-1)
                        index += 1

            # store (s,a,r,s') into replay buffer
            for i in range(L): # each UAV i

                # reward = alphaL*(Rwd(s, a) + r_*max(a')Q(s', a'))
                # where maxQ = max(a')Q(s', a')
                maxQ = dq.maxQ(oldS[i], action_list[i], t, i, k, R, actionSpace, clusters, B, PU, g, l_, o2)
                reward = alphaL * (directReward_list[i] + r_ * maxQ)
                
                replayBuffer.append([oldS[i], action_list[i], reward, newS[i]])
            
            # Randomly select a minibatch of H_ samples from replay buffer
            H_ = min(30, len(replayBuffer)) # select maximum 30 samples
            
            minibatch = [] # minibatch of H_ samples from replay buffer
            
            while len(minibatch) < H_:
                rand = random.randint(0, len(minibatch)-1) # randomly select
                minibatch.append(replayBuffer[rand]) # append to the buffer
            
            # train the network and update weight
            deepLearningQ_training(QTable, 'cpu:0', 50, False)

if __name__ == '__main__':
    
    # parameters (as TABLE 1)
    fc = 800000000 # carrier frequency
    B = 1000000 # bandwidth
    o2 = -110 # Noise power spectral
    b1 = 0.36 # environmental parameter
    b2 = 0.21 # environmental parameter
    alpha = 2 # path loss exponent
    u1 = 3 # additional path loss for LoS
    u2 = 23 # additional path loss for NLoS
    alphaL = 0.0001 # learning rate for DQN
    r_ = 0.7 # discount factor

    width = 50 # width (m)
    height = 50 # height (m)

    M = 1000 # M = 1000 episodes
    L = 3 # L = 3 clusters = 3 UAVs
    devices = 50 # 50 devices
    T = 10 # T = 10s
    H = 15 # H = 15m

    # run
    algorithm1(M, T, L, devices, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_)
