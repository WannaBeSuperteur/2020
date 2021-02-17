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
def updateDRlist(UAVs, value, i, deviceList, b1, b2, S_, u1, u2, fc, n, action, a,
                 Q, s_i, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g):

    # for each device k
    for k in range(len(clusters[i])):

        # get PLoS and PNLoS
        PLoS_i = f.getPLoS(False, n, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), b1, b2, S_)
        PNLoS_i = f.getPLoS(True, n, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), b1, b2, S_)
            
        # update Q value                
        g_i = f.g_nlkl(PLoS_i, u1, PNLoS_i, u2, fc, n, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), alpha)

        dq.updateQvalue(Q, s_i, action, a, value, alpha, r_, n, UAVs, i, R, useDL, clusters, B, PU, g, I_, o2)
        directReward_list[i] += value

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
def algorithm1(M, T, L, devices, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_, S_):

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
    # a[n][l][k_l] (a[n][l][k])  : the number of times that each device communicates with UAV l
    # g[n][l][k_l] (g[n][l][k])  : the channel's power gain between UAV l and device k_l
    # R[n][k_l]    (R[n][l][k])  : the average throughput of devices (for each device k),
    #                              in l-th cluster (1d array, for the devices in l-th cluster)
    # PU[n][k_l]   (PU[n][l][k]) : peak power for IoT devices' uplink transmit (at most -20dBm -> assumption: -20dBm)
    # I_[n][k_l]   (I_[n][l][k]) : inference received by UAV l (assumption: 0 as default)
    a = []
    g = []
    PU = []
    R = []
    I_ = []
    
    for t in range(T + 1): # n = 0,1,...,T (# of time slots)

        # not n = 0,1,...,T-1 to prevent bug
        temp_a = []
        temp_g = []
        temp_PU = []
        temp_R = []
        temp_I = []
        
        for l in range(L): # l = 0,1,...,L-1 (# of clusters = # of UAVs)
            temp_a_ = []
            temp_g_ = []
            temp_PU_ = []
            temp_R_ = []
            temp_I_ = []

            K = len(clusters[l]) # number of devices for UAV l

            for k in range(K): # k = 0,...,K-1 (# of devices for each UAV l)
                temp_a_.append(0)
                temp_g_.append(0)
                temp_PU_.append(-20)
                temp_R_.append(0)
                temp_I_.append(0)

            temp_a.append(temp_a_)
            temp_g.append(temp_g_)
            temp_PU.append(temp_PU_)
            temp_R.append(temp_R_)
            temp_I.append(temp_I_)
                
        a.append(temp_a)
        g.append(temp_g)
        PU.append(temp_PU)
        R.append(temp_R)
        I_.append(temp_I)

    # init Q Table
    # Q Table = [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    Q = []

    ### TRAIN ###
    replayBuffer = [] # REPLAY BUFFER
    
    for episode in range(M):
        for t in range(T): # each time slot t

            # use deep learning if length of Q table >= 100
            if len(Q) >= 100: useDL = True
            else: useDL = False

            # initialize direct reward
            directReward_list = [] # direct reward for action (for each UAV)
            for i in range(L): directReward_list.append(0) # initialize as 0
            
            oldS_list = [] # old states (for each UAV)
            action_list = [] # actions (for each UAV)
            newS_list = [] # new states (for each UAV)
            
            for i in range(L): # for each UAV = each cluster
                
                # get UAV i's next location
                s_i = dq.getS(UAVs[i], t, i, a, R)

                # choose action with e-greedy while e increases
                e_ = episode / M # example
                action = dq.getActionWithE(Q, s_i, e_)
                
                nextLocation = dq.getNextLocation(s_i, action, t, UAVs, l, a, R)

                # if UAV i files beyond the border
                if beyondBorder(UAVs[i], t, width, height) == True:
                    
                    # UAV i stays at the border
                    if UAVs[i][0][t] < 0: UAVs[i][0][t] = 0 # x value < 0
                    elif UAVs[i][0][t] > width: UAVs[i][0][t] = width # x value > width
                    
                    if UAVs[i][1][t] < 0: UAVs[i][1][t] = 0 # y value < 0
                    elif UAVs[i][1][t] > height: UAVs[i][1][t] = height # y value > height
                    
                    # UAV i gets a penalty of -1
                    updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, s_i, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g)

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
                        s_i = dq.getS(UAVs[i], t, i, a, R)
                        updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, s_i, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g)

                        s_j = dq.getS(UAVs[j], t, j, a, R)
                        updateDRlist(UAVs, -1, j, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, s_j, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g)

                # get throughput (before) (time = n)
                beforeThroughput = f.R_nkl(B, t, i, t, PU, g, I_, o2)
                
                # get and move to next state (update q, a and R)
                # s       : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
                # q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
                # action  : action ([-1, -1, -1] to [1, 1, 1])

                # "Don't be confused between s = [q, a, R] and [state, action, newState]"
                
                s = dq.getS(UAVs[i], t, i, a, R) # current state                
                oldS = copy.deepcopy(s) # save old state

                (nextState, _) = dq.getNextState(s, action, t, UAVs, i, a, R, clusters, B, PU, g, I_, o2)
                q_next = copy.deepcopy(nextState[0])
                a_next = copy.deepcopy(nextState[1])
                R_next = copy.deepcopy(nextState[2])
                s = [q_next, a_next, R_next] # update current state

                # append to oldS_list and newS_list
                oldS_list.append(oldS)
                newS_list.append(s)
                action_list.append(action)

                # get throughput (after) (time = t+1)
                afterThroughput = f.R_nkl(B, t, i, t+1, PU, g, I_, o2)

                # device t's throughput does not increase
                # assumption: device t is the device communicated with when the time slot is t
                # [temporary] assume that the index of the device is t
                if afterThroughput <= beforeThroughput:
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAVs[i], t, i, a, R)
                    updateDRlist(UAVs, -1, i, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, s_i, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g)

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
                        s_UAV = dq.getS(UAV, t, index, a, R)
                        updateDRlist(UAVs, -1, index, deviceList, b1, b2, S_, u1, u2, fc, t, action, a,
                                 Q, s_UAV, alpha, r_, R, useDL, clusters, B, PU, I_, o2, directReward_list, g)
                        index += 1

            # store (s,a,r,s') into replay buffer
            for i in range(L): # each UAV i

                # for each device k,
                # reward = alphaL*(Rwd(s, a) + r_*max(a')Q(s', a'))
                # where maxQ = max(a')Q(s', a')
                maxQs = []
                rewards = []

                # for each device k
                for k in range(len(clusters[i])):
                    PLoS_i = f.getPLoS(False, t, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), b1, b2, S_)
                    PNLoS_i = f.getPLoS(True, t, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), b1, b2, S_)
                    
                    g_i = f.g_nlkl(PLoS_i, u1, PNLoS_i, u2, fc, t, i, k, clusters, x(UAVs), y(UAVs), h(UAVs), alpha)
                    maxQs.append(dq.getMaxQ(oldS_list[i], action_list[i], t, UAVs, i, a, R, actionSpace, clusters, B, PU, g_i, I_, o2, useDL))
                    rewards.append(alphaL * (directReward_list[i] + r_ * maxQs[k]))

                    # append to Q Table: [[[s0], [Q00, Q01, ...]], [[s1], [Q10, Q11, ...]], ...]
                    # where s = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
                    # and   Q = reward
                    # from oldS_list, action_listddd and reward
                    action_rewards = copy.deepcopy(zero_27)
                    action_rewards[dq.getActionIndex(action_list[i])] = rewards[k]
                    
                    QTable.append([oldS_list[i], action_rewards, i, k])

                    # append to replay buffer
                    replayBuffer.append([oldS_list[i], action_list[i], rewards[k], newS_list[i]])
            
            # Randomly select a minibatch of H_ samples from replay buffer
            H_ = min(30, len(replayBuffer)) # select maximum 30 samples
            
            minibatch = [] # minibatch of H_ samples from replay buffer

            while len(minibatch) < H_:
                rand = random.randint(0, len(replayBuffer)-1) # randomly select from the replay buffer
                minibatch.append(replayBuffer[rand]) # append to the buffer
            
            # train the network and update weight
            dq.deepLearningQ_training(QTable, 'cpu:0', 50, False)

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
    S_ = 0.0 # (temp) constant value determined by both the antenna and the environment
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
    algorithm1(M, T, L, devices, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, S_, alphaL, r_)
