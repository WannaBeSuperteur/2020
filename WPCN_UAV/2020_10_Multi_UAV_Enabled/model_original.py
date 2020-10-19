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
from shapely.geometry import LineString

# PAPER : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1

# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is the vector of location, from previous t to current t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...

# check if UAV i files beyound the border
def beyondBorder(UAVi, t, border):
    # ( [0] check if UAV i files beyond the border )
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

# check if minimum throughput of all devices in a cluster == 0
def isMinThroughputOfAllDevicesInCluster0(t):
    # ( [2] check )
    return True

# check if minimum throughput of all devices does not increase
def isMinThroughputOfAllDevicesDoesNotInc(t):
    # ( [3] check )
    return True

# ALGORITHM 1
# 3D trajectory design and time resource allocation solution based on DQL
# M                  : number of episodes
# T                  : number of time slots
# Litalic            : number of clusters
# L                  : number of UAVs
# width, height      : width and height of the board
# fc, B, o2, b1, b2, : parameters (as Table 1)
# alpha, u1, u2,
# alphaL, r
def algorithm1(M, T, Litalic, L, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r):

    # Q Table: [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    QTable = [] # Q table

    # info about all UAVs : [UAV0, UAV1, ...] = [[x0, y0, h0], [x1, y1, h1], ...]
    UAVs = []

    # randomly place UAVs
    # (    )

    # cluster UAVs using K-means clustering
    # (    )

    # ( [4] init target network and online network )
    # ( [5] init UAV's location and IoT devices' location )
    # ( [6] init border )

    for episode in range(1, M+1):
        for t in range(1, T+1): # each time slot
            
            for i in range(1, L+1): # each UAV

                # ( [7] choose action with e-greedy while e increases )
                # ( [8] get UAV i's next location )

                # if UAV i files beyond the border
                if beyondBorder(UAV[i], t, border) == True:
                    # ( [9] UAV i stays at the border )
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAV[i], q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

            for i in range(1, L+1): # each UAV i
                for j in range(1, i): # each UAV j

                    # UAV i and j's trajectory exists cross
                    if IsTrajectoryCrossed(UAV[i], UAV[j], t) == True:
                        # ( [11] UAV i and UAV j stay at the previous location )

                        # UAV i and UAV j get a penalty of -1
                        s_i = dq.getS(UAV[i], q, n, l, a, k, R)
                        dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

                        s_j = dq.getS(UAV[j], q, n, l, a, k, R)
                        dq.updateQvalue(Q, s_j, a, -1, alpha, lb, q, n, l, k, R, useDL)

                beforeThroughput = 0 # ( [13] get throughput )

                # ( [14] execute action )
                # ( [15] get next state )

                afterThroughput = 0 # ( [16] get throughput )

                # device t's throughput does not increase
                if afterThroughput <= beforeThroughput:
                    
                    # UAV i gets a penalty of -1
                    s_i = dq.getS(UAV[i], q, n, l, a, k, R)
                    dq.updateQvalue(Q, s_i, a, -1, alpha, lb, q, n, l, k, R, useDL)

            # if time slot is T
            if t == T:
                # if minimum throughput of all devices in a cluster == 0
                if isMinThroughputOfAllDevicesInCluster0(t) == True:

                    # ( [18] The UAV get a penalty of -2 )

                # if minimum throughput of all devices does not increase
                if isMinThroughputOfAllDevicesDoesNotInc(t) == True:

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

    width = 50 # width (m)
    height = 50 # height (m)

    M = 1000 # M = 1000 episodes
    Litalic = 3 # Litalic = 3 clusters
    L = 50 # L = 50 UAVs
    T = 1 # T = 1s
    H = 15 # H = 15m

    # run
    algorithm1(M, T, Litalic, L, width, height, H, fc, B, o2, b1, b2, alpha, u1, u2, alphaL, r_):
