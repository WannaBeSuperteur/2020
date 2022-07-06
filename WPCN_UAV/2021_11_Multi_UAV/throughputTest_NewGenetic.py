import sys
import os

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
import algorithm_base as algo_base

import time

#### initialize the movement of UAV ####
def initializeMovementOfUAV(devices):
    return list(np.random.permutation(range(devices)))

#### compute direction list ####
def computeDirectionList(bestParams, q, l, N, deviceListC, initialMovement, width, height):

    param1 = bestParams[0]
    param2 = bestParams[1]

    UAV_x = q[l * (N+1)][2]
    UAV_y = q[l * (N+1)][3]
    UAV_h = q[l * (N+1)][4]
    initialLocUAV = [UAV_x, UAV_y, UAV_h]
            
    (_, _, directionList) = findOptimalPath(N, deviceListC, initialLocUAV, initialMovement, param1, param2, width, height)
    return directionList

# genetic-like algorithm to decide optimal path (= direction list)

#### NOTE ####
# movement     -> device visit sequence
# path         -> direction list = list of directions of UAV to move
# optimal path -> path of the optimal movement with the formula to minimize

# minimum of A*(total movement distance) + B*(sum of 1/d^2 by moving minimum times)
# parameter 1 -> 5 * (random swap probability of two neighboring device)
# parameter 2 -> proportion of A and B
def findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, param2, width, height, printed=False):

    # define parameter A and B using param2 (between 0.0 ~ 1.0)
    A = param2
    B = 1.0 - param2
    moves = len(initialMovement)

    currentBestScore    = computeScore(A, B, initialLocUAV, initialMovement, deviceList, width, height)
    currentBestMovement = initialMovement

    # 30 iterations
    for i in range(30):
        bestModifiedMovement = None
        
        # create 20 candidate moves in an iteration to find ONLY ONE best move
        for j in range(20):
            ###print(i, j)
            modifiedMovement = copy.deepcopy(currentBestMovement)

            # change the movement
            for j in range(moves-1):
                r = random.random()

                # random swap with probability (param1)
                if r < param1 / 5.0:
                    modifiedMovement[j], modifiedMovement[j+1] = modifiedMovement[j+1], modifiedMovement[j]

            # compute the score
            score = computeScore(A, B, initialLocUAV, modifiedMovement, deviceList, width, height)

            # update best modified score/movement
            if bestModifiedMovement == None or score < bestModifiedScore:
                bestModifiedScore    = score
                bestModifiedMovement = copy.deepcopy(modifiedMovement)

        # update current best score and movement at the end of each iteration
        currentBestScore    = bestModifiedScore
        currentBestMovement = bestModifiedMovement

    # define best movement as the current best movement at the end of all the iterations
    bestMovement = currentBestMovement

    # create the optimal path based on the best movement
    (locsUAV, optimalPath) = createOptimalPath(N, initialLocUAV, bestMovement, deviceList, width, height, printed)

    return (locsUAV, bestMovement, optimalPath)

# distance between UAV and device (also can be between device A and device B)
def dist(locUAV, locDevice):
    if len(locUAV) == 3:
        return math.sqrt(pow(locUAV[0] - locDevice[0], 2) + pow(locUAV[1] - locDevice[1], 2) + pow(locUAV[2], 2))
    else:
        return math.sqrt(pow(locUAV[0] - locDevice[0], 2) + pow(locUAV[1] - locDevice[1], 2))

# compute the path (with moving minimum times) corresponding to the movement
def computeMinimumPath(initialLocUAV, movement, deviceList, width, height):

    # list of possible unit movement
    unitMovements = []
    for i in range(8):
        unitMovements.append([5.0 * math.cos(i * math.pi / 4), 5.0 * math.sin(i * math.pi / 4)])
    unitMovements.append([0.0, 0.0])

    # use only X and Y location of UAV
    minimumPath = [] # array of moving directions of UAV
    locsUAV     = [] # locations of UAV after each move in the path
    stops       = [] # the indices of stop points, in minimumPath

    # initialize the location of UAV
    currentLocUAV = initialLocUAV[:2]

    # make path (DO NOT consider N, the number of time slots)
    for d in range(len(deviceList)):

        # find the index of device with the given movement array
        deviceIndex = movement[d]
        device = deviceList[deviceIndex]

        # move toward best direction until "do not move" becomes closest to the device
        while True:
            minDist          = dist(currentLocUAV, device)
            minDistDirection = 8

            # find the index of the best direction (index 0~7: move, index 8: stop)
            for i in range(8):
                nextLocUAV    = np.array(currentLocUAV) + np.array(unitMovements[i])
                nextLocUAV    = list(nextLocUAV)
                nextLocUAV[0] = np.clip(nextLocUAV[0], 0, width) # limit x value
                nextLocUAV[1] = np.clip(nextLocUAV[1], 0, height) # limit y value
                nextDist      = dist(nextLocUAV, device)

                if nextDist < minDist:
                    minDist          = nextDist
                    minDistDirection = i

            ###print(d, np.array(currentLocUAV), np.array(unitMovements[minDistDirection]), np.round_(minDist, 6), minDistDirection)

            # stop is best for the UAV
            if minDistDirection == 8:
                stops.append(len(minimumPath))
                break

            # add movement and update the location of UAV (with clipping)
            else:
                tempCurrentLocUAV = np.array(currentLocUAV)
                
                currentLocUAV = np.array(currentLocUAV) + np.array(unitMovements[minDistDirection])
                currentLocUAV[0] = np.clip(currentLocUAV[0], 0.0, width) # limit x value
                currentLocUAV[1] = np.clip(currentLocUAV[1], 0.0, height) # limit y value

                # clipping result is the same with original -> not meaningful movement
                if dist(currentLocUAV, tempCurrentLocUAV) < 1e-6:
                    stops.append(len(minimumPath))
                    break

                # meaningful movement
                else:
                    currentLocUAV = list(currentLocUAV)

                    minimumPath.append(minDistDirection)
                    locsUAV.append(currentLocUAV)

    return (locsUAV, minimumPath, stops)
    
# create the optimal path based on the best movement
def createOptimalPath(N, initialLocUAV, bestMovement, deviceList, width, height, printed=False):

    # suppose that the UAV "stops" near each device for a while
    
    # compute the time needed to move as bestMovement
    # (suppose that it is below N, otherwise common throughput can be zero)
    (locsUAV, minimumPath, closeness, stops) = computeMinimumPathAndClosenessWithMinMoves(initialLocUAV, bestMovement, deviceList, width, height)
    bestMoveTime = len(minimumPath) + 1

    # compute the remaining time ( = N - (minimum path length) )
    remainingTime = (N + 1) - bestMoveTime

    # find (dist^2) using closeness
    sqDistList = []
    for c in closeness:
        sqDistList.append(1 / max(1e-6, c))

    # the sum of sqDistList -> 1
    sqDistListNormalized = np.array(sqDistList) / np.sum(sqDistList)

    if printed:
        print('\n closeness :')
        print(closeness)

        print('\n sqDistListNormalized :')
        print(sqDistListNormalized)

    # add random small noise to sqDistListNormalized
    for i in range(len(sqDistListNormalized)):
        sqDistListNormalized[i] += random.random() * 1.0e-8

    if printed:
        print('\n sqDistListNormalized with noise :')
        print(sqDistListNormalized)

    # visit sequence based on the best movement
    sqDistListNormalizedVisit = []
    for i in range(len(bestMovement)):
        sqDistListNormalizedVisit.append(sqDistListNormalized[bestMovement[i]])

    # change to NumPy array
    sqDistListNormalizedVisit = np.array(sqDistListNormalizedVisit)

    if printed:
        print('\n sqDistListNormalized with best movement visit sequence :')
        print(sqDistListNormalizedVisit)

    # allocate the remaining time for each device in proportion to (dist)^2
    # (maybe use binary search algorithm)
    remainingTimeForEachDevice = []

    minWeight = 0.0
    maxWeight = N * len(deviceList)
    fitWeight = None

    # use remaining time for stop
    if remainingTime > 0:
        
        while True:
            midWeight = (minWeight + maxWeight) / 2.0
            remainingTimeDistribution = np.array(sqDistListNormalizedVisit * midWeight).astype(int)

            if sum(remainingTimeDistribution) > remainingTime:
                maxWeight = midWeight
            elif sum(remainingTimeDistribution) < remainingTime:
                minWeight = midWeight
            else:
                fitWeight = midWeight
                remainingTimeForEachDevice = remainingTimeDistribution
                break

        # append "stop"s to the optimal path based on the allocated remaining time for each device
        remainingReverse = remainingTimeForEachDevice[::-1]
        
        for i in range(len(stops)):
            stop = stops[len(stops) - (i+1)]
            stopCount = remainingReverse[i]
            minimumPath = minimumPath[:stop] + [8 for j in range(stopCount)] + minimumPath[stop:]

            if stop > 0:
                locsUAV = locsUAV[:stop] + [locsUAV[stop-1] for j in range(stopCount)] + locsUAV[stop:]
            else:
                locsUAV = [initialLocUAV[:2] for j in range(stopCount)] + locsUAV[stop:]

        # return the list of the locations of UAV + optimal path (=minimumPath)
        return (locsUAV, minimumPath)

    else:
        
        # NO REMAINING TIME
        # return first N elements of the list of the locations of UAV + minimum path
        return (locsUAV[:N], minimumPath[:N])

# optimal : minimize A*(total movement distance) + B*(sum of 1/(dist)^2 by moving minimum times)
def computeScore(A, B, initialLocUAV, movement, deviceList, width, height):
    A_score = A * computeTotalDist(initialLocUAV, movement, deviceList)
    (_, _, closeness, _) = computeMinimumPathAndClosenessWithMinMoves(initialLocUAV, movement, deviceList, width, height)
    B_score = B * sum(closeness)
    return A_score + B_score

# compute (total movement distance, NOT CONSIDERING THE MOVEMENT OF UAV, just Euclidean)
def computeTotalDist(initialLocUAV, movement, deviceList):
    
    # initial UAV location <-> first device
    totalDist = dist(initialLocUAV, deviceList[movement[0]])

    # k-th device <-> (k+1)-th device
    for i in range(len(deviceList)-1):
        totalDist += dist(deviceList[movement[i]], deviceList[movement[i+1]])

    return totalDist

# compute (sum of 1/d^2 by moving minimum times) where d = distance
def computeMinimumPathAndClosenessWithMinMoves(initialLocUAV, movement, deviceList, width, height):
    
    # compute the minimum path first
    (locsUAV, minimumPath, stops) = computeMinimumPath(initialLocUAV, movement, deviceList, width, height)

    # compute the closeness for each device
    closeness = [0.0 for i in range(len(deviceList))]

    for i in range(len(minimumPath)): # except for (first element of the minimum path = original UAV location)
        for j in range(len(deviceList)):
            closeness[j] = max(closeness[j], 1.0 / pow(dist(locsUAV[i], deviceList[j]), 2))

    return (locsUAV, minimumPath, closeness, stops)

# return the list of device in the current cluster
def findDevicesInCluster(l, deviceList, cluster_mem):
    newDeviceList = []

    for i in range(len(deviceList)):
        if cluster_mem[i] == l:
            newDeviceList.append(deviceList[i])

    return newDeviceList

#### find device location ####
def getDeviceLocation(l, w, final_throughputs):
    pass

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=170)

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
                                'iters':'int', 'clusteringAtLeast':'float', 'clusteringAtMost':'float',
                                'windowSize':'int', 'epochs':'int'})

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
    clusteringAtLeast = paperArgs['clusteringAtLeast']
    clusteringAtMost = paperArgs['clusteringAtMost']
    windowSize = paperArgs['windowSize']
    epochs = paperArgs['epochs']

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
    configContent += 'iters=' + str(iters) + '\n'
    #configContent += 'windowSize=' + str(WINDOWSIZE) + '\n'
    configContent += 'clusteringAtLeast=' + str(clusteringAtLeast) + '\n'
    configContent += 'clusteringAtMost=' + str(clusteringAtMost) + '\n'
    configContent += 'windowSize=' + str(windowSize) + '\n'
    configContent += 'epochs=' + str(epochs)

    configFile.write(configContent)
    configFile.close()

    # minimum throughput list
    minThroughputList = []

    # run for static (UAV location fixed : above the centroid of its cluster)
    # ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047 (reference [1] of the paper)
    static_input_data = []
    static_output_data = []

    for iterationCount in range(iters // 2):
        print('STATIC ITER COUNT ', iterationCount, '/', iters // 2)

        algo_base.throughputTest(M, T, N, L, devices, width, height, H,
                                 ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                                 iterationCount, iters, minThroughputList, clusteringAtLeast, clusteringAtMost,
                                 static_input_data, static_output_data, True, None, windowSize, True,
                                 base_func_initializeMovementOfUAV=initializeMovementOfUAV,
                                 base_func_computeDirectionList=computeDirectionList,
                                 base_func_getDeviceLocation=None)

    # re-initialize minimum throughput list
    minThroughputList = []

    # run training
    try:
        _ = pd.read_csv('train_input_preprocessed.csv' , index_col=0)
        _ = pd.read_csv('train_output_preprocessed.csv', index_col=0)

    except:
        print('[ NO PREPROCESSED DATA ]')

        # input and output data for deep learning
        input_data = []  # input data  (parameters of the algorithm)
        output_data = [] # output data (minimum throughput values)

        for iterationCount in range(iters):
            print('TRAINING ITER COUNT ', iterationCount, '/', iters)

            algo_base.throughputTest(M, T, N, L, devices, width, height, H,
                                     ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                                     iterationCount, iters, minThroughputList, clusteringAtLeast, clusteringAtMost,
                                     input_data, output_data, True, None, windowSize, False,
                                     base_func_initializeMovementOfUAV=initializeMovementOfUAV,
                                     base_func_computeDirectionList=computeDirectionList,
                                     base_func_getDeviceLocation=None)

    # get and train model
    try:
        model = tf.keras.models.load_model('WPCN_UAV_DL_model')
    except:
        print('model load failed')
        model = algo_base.getAndTrainModel(epochs, windowSize)

    # find the best genetic algorithm argument based on the WPCN UAV network environment arguments

    # run test (using 5% (min 10) iterations of training)
    iters = max(10, iters // 20)

    # re-initialize minimum throughput list
    minThroughputList = []

    # input and output data for testing
    test_input_data = []
    test_output_data = []

    for iterationCount in range(iters):
        print('TEST ITER COUNT ', iterationCount, '/', iters)

        algo_base.throughputTest(M, T, N, L, devices, width, height, H,
                                 ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                                 iterationCount, iters, minThroughputList, clusteringAtLeast, clusteringAtMost,
                                 test_input_data, test_output_data, False, model, windowSize, False,
                                 base_func_initializeMovementOfUAV=initializeMovementOfUAV,
                                 base_func_computeDirectionList=computeDirectionList,
                                 base_func_getDeviceLocation=None)
