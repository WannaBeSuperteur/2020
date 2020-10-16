import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main

import formula as f
import deepQ as dq

import math
import random
import numpy as np

# PAPER : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1

# check if UAV i files beyound the border
def beyondBorder(UAV_i, border):
    # ( [0] check if UAV i files beyond the border )
    return True

# check if UAV i's trajectory and UAV j's trajectory is crossed
def IsTrajectoryCrossed(UAV_i, UAV_j):
    # ( [1] check if UAV i's trajectory and UAV j's trajectory is crossed )
    return True

# check if minimum throughput of all devices in a cluster == 0
def isMinThroughputOfAllDevicesInCluster0(device):
    # ( [2] check )
    return True

# check if minimum throughput of all devices does not increase
def isMinThroughputOfAllDevicesDoesNotInc():
    # ( [3] check )
    return True

# ALGORITHM 1
# 3D trajectory design and time resource allocation solution based on DQL
# M                  : number of episodes
# T                  : number of time slots
# L                  : number of UAVs
# fc, B, o2, b1, b2, : parameters (as Table 1)
# alpha, u1, u2,
# alphaL, r
def algorithm1(M, T, L, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r):

    # Q Table: [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    QTable = [] # Q table
    UAVs = [] # info about all UAVs

    # ( [4] init target network and online network )
    # ( [5] init UAV's location and IoT devices' location )
    # ( [6] init border )

    for episode in range(1, M+1):
        for t in range(1, T+1): # each time slot
            
            for i in range(1, L+1): # each UAV

                # ( [7] choose action with e-greedy while e increases )
                # ( [8] get UAV i's next location )

                # if UAV i files beyond the border
                if beyondBorder(UAV_i, border) == True:
                    # ( [9] UAV i stays at the border )
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAV_i, q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

            for i in range(1, L+1): # each UAV

                # UAV i and j's trajectory exists cross
                if IsTrajectoryCrossed(UAV_i, UAV_j) == True:
                    # ( [11] UAV i and UAV j stay at the previous location )

                    # UAV i and UAV j get a penalty of -1
                    s_i = dq.getS(UAV_i, q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

                    s_j = dq.getS(UAV_j, q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_j, a, -1, alpha, lb, q, n, l, k, R, useDL)

                beforeThroughput = 0 # ( [13] get throughput )

                # ( [14] execute action )
                # ( [15] get next state )

                afterThroughput = 0 # ( [16] get throughput )

                # device t's throughput does not increase
                if afterThroughput <= beforeThroughput:
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAV_i, q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

            # if time slot is T
            if t == T:
                # if minimum throughput of all devices in a cluster == 0
                if isMinThroughputOfAllDevicesInCluster0(t) == True:

                    # ( [18] The UAV get a penalty of -2 )

                # if minimum throughput of all devices does not increase
                if isMinThroughputOfAllDevicesDoesNotInc == True:

                    # All UAVs get a penalty of -1
                    for UAV in UAVs:
                        s_UAV = dq.getS(UAV, q, n, l, a, k, R)
                        dq.updateQvalue(Q, s_UAV, a, -1, alpha, lb, q, n, l, k, R, useDL)

            # ( [20] store (s,a,r,s') into replay buffer )
            # ( [21] Randomly select a minibatch of H samples from replay buffer )
            # ( [22] Train the network and update weight )

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

    M = 1000 # M = 1000 episodes
    L = 3 # L = 3 UAVs
    T = 1 # T = 1s
    H = 15 # H = 15m

    # run
    algorithm1(M, T, L, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_):
