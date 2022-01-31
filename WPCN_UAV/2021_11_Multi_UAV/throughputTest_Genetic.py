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

import time

timeCheck = h_.loadSettings({'timeCheck':'logical'})['timeCheck']
printDetails = h_.loadSettings({'printDetails':'logical'})['printDetails']

# enable GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# allow GPU memory increase
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# window size
WINDOWSIZE = h_.loadSettings({'windowSize':'int'})['windowSize']

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

# move UAV using direction info
#  x       mod 3 == 0 -> turn left, 1 -> straight, 2 -> turn right
# (x // 3) mod 3 == 0 -> forward  , 1 -> hold    , 2 -> backward
# (x // 9) mod 3 == 0 -> h+1      , 1 -> h       , 2 -> h-1

# initial direction: positive direction of x-axis (0 degree)
# q = [[l, t, xlt, ylt, hlt], ...] -> update q
def moveUAV(q, directionList, N, l, width, height):

    degree = 0

    for t in range(N):

        direction = directionList[t]

        # check if None value
        if direction == None: continue

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

        # update corresponding q value as new q
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

# define and train model
def defineAndTrainModel(train_input, train_output, test_input, test_output, epochs):
    model = DEEP_LEARNING_MODEL(window=WINDOWSIZE)

    # training setting (early stopping and reduce learning rate)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             mode="min",
                                             patience=5)

    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=2,
                                                      verbose=1,
                                                      min_delta=0.0001,
                                                      mode='min')

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train using input and output data
    model.fit(train_input, train_output,
              validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)
        
    model.summary()
    model.save('WPCN_DL_model')

    # test the model (validation)
    test_prediction = model.predict(test_input)
    test_prediction = np.reshape(test_prediction, (len(test_prediction), 1))
    test_result     = np.concatenate((test_prediction, test_output), axis=1)

    test_result     = pd.DataFrame(test_result)
    test_result.to_csv('train_valid_result.csv')

    # return the trained model
    return model

# deep learning model
def getAndTrainModel(epochs):

    # load and preprocess input and output data
    try:
        input_data  = pd.read_csv('input_data_preprocessed.csv' , index_col=0)
        output_data = pd.read_csv('output_data_preprocessed.csv', index_col=0)
    except:
        print('[ NO PREPROCESSED DATA ]')
        (input_data, output_data) = preprocessData()

    # convert data into numpy array
    input_data  = np.array(input_data)
    output_data = np.array(output_data)

    # divide into training and test data
    toTest         = 0.1
    testStartIndex = int(len(input_data) * (1.0 - toTest))
    
    train_input  = input_data[:testStartIndex]
    train_output = output_data[:testStartIndex]
    test_input   = input_data[testStartIndex:]
    test_output  = output_data[testStartIndex:]

    # model definition (with regularizer)
    try:
        with tf.device('/gpu:0'):
            return defineAndTrainModel(train_input, train_output, test_input, test_output, epochs)
    except:
        print('GPU load failed -> using CPU')
        with tf.device('/cpu:0'):
            return defineAndTrainModel(train_input, train_output, test_input, test_output, epochs)

# save trajectory as graph
def saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, q, markerColors, training):

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
    if training == True:
        plt.savefig('train_trajectory_iter' + ('%04d' % iterationCount))
    else:
        plt.savefig('test_trajectory_iter' + ('%04d' % iterationCount))

# update a_l,kl[n]
# alkl      : a_l,kl[n] for each UAV l, device k and time slot n
#             where alkl = [[l0, k, l1, n, value], ...
def update_alkl(alkl, q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs, devices):

    # decide the device to communicate with
    deviceToCommunicate = algo.findDeviceToCommunicate(q, w, l, t, N, s, b1, b2,
                                                       mu1, mu2, fc, c, alphaP, numOfDevs)

    # update alkl, in the form of [[l0, k, l1, n, value], ...]
    alkl_index = f.find_alkl(alkl, l, deviceToCommunicate, l, t)
            
    # when DO NOT have alkl with (l, deviceToCommunicate, l, t)
    if alkl_index == None or alkl_index == len(alkl):
        alkl.append([l, deviceToCommunicate, l, t, 1])

        # set alkl as 0 for other devices
        for k in range(devices):
            if k == deviceToCommunicate: continue
            alkl.append([l, k, l, t, 0])

    # when have alkl with (l, deviceToCommunicate, l, t)
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

# compute throughput value for the direction list
def getThroughput(alkl, q, w, l, N, T, s, b1, b2, mu1, mu2, fc, c,
                  alphaP, PU, numOfDevs, devices, printDetails,
                  directionList, width, height):

    # move UAV from time from 0 to T (N+1 times, N moves), for all UAVs of all clusters
    # (update q)
    moveUAV(q, directionList, N, l, width, height)

    # throughput results
    throughputs = []

    for t in range(N):
        if printDetails == True:
            print('cluster ' + str(l) + ' / ' + str(L) + ', time ' + str(t) + ' / ' + str(N))

        # update a_l,kl[n] for this (l, t)
        update_alkl(alkl, q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs, devices)

        # compute average throughput for each device in L
        thrputs = []

        for k in range(devices):
            thrput = f.formula_11(q, w, l, k, alphaP, N, T, s, b1, b2, mu1, mu2, fc, c, L, alkl, PU, numOfDevs)[-1]
            thrputs.append(thrput)

            if printDetails == True:
                print('l=' + str(l) + ' k=' + str(k) + ' throughput=' + str(thrput))

        throughputs.append(thrputs)

    # throughputs: shape (k, N) -> shape (N, k)
    throughputs = np.array(throughputs)
    final_throughputs = throughputs[-1]

    return (throughputs, final_throughputs)

# running throughput test
def throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU,
                   iterationCount, minThroughputList, clusteringAtLeast):

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
        (q, w, cluster_mem, markerColors) = algo.kMeansClustering(L, deviceList, width, height, H, N, False, True, False)
        numOfDevs = [cluster_mem.count(l) for l in range(L)]
        if min(numOfDevs) >= int(clusteringAtLeast * devices // L): break

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

        # the number of devices in cluster l
        devices = numOfDevs[l]
        directionList = [None for i in range(N)]

        # make direction list using random (when training)
        for t in range(N):
            directionList[t] = random.randint(0, 3*3*3-1)

        # modifying directions for minimum (common) throughput maximization
        rounds = 50
        for round in range(rounds):
            print(round)

            # get throughput
            (throughputs, final_throughputs) = getThroughput(alkl, q, w, l, N, T, s, b1, b2, mu1, mu2, fc, c,
                                                             alphaP, PU, numOfDevs, devices, printDetails,
                                                             directionList, width, height)

            print('min throughputs:', min(final_throughputs))

            # terminate here at the last round
            if round == rounds - 1: break

            minThroughput = min(final_throughputs) # min throughput
            zeros         = list(final_throughputs).count(0) # number of zero throughputs
            avgThroughput = np.mean(final_throughputs) # average throughput

            # find near cases (and select the best 2 near cases) with change probability p
            p = 0.1 - 0.0015 * round
            print('probability:', p)
            nearCasesInfo = []

            # find 30 near cases in total
            for nearCaseNo in range(30):
                nearCase = copy.deepcopy(directionList)

                # randomly modify each element of direction list
                for i in range(len(nearCase)):
                    if random.random() < p:
                        nearCase[i] = random.randint(0, 3*3*3-1)

                # throughput test using modified direction list
                (_, final_near) = getThroughput(alkl, q, w, l, N, T, s, b1, b2, mu1, mu2, fc, c,
                                                alphaP, PU, numOfDevs, devices, printDetails,
                                                nearCase, width, height)

                nearCasesInfo.append([min(final_near),
                                      list(final_near).count(0),
                                      np.mean(final_near),
                                      nearCase])

                print(min(final_near),
                      list(final_near).count(0),
                      np.mean(final_near),
                      nearCase)

            # find the best near cases (order: min -> number of zeros -> average throughput)
            nearCasesInfo.sort(key=lambda x: x[2], reverse=True) # sort by average
            nearCasesInfo.sort(key=lambda x: x[1], reverse=True) # sort by zeros
            nearCasesInfo.sort(key=lambda x: x[0], reverse=True) # sort by min

            directionList = nearCasesInfo[0][3]

        # save throughputs at first iteration
        if iterationCount == 0:
            throughputs_df = pd.DataFrame(throughputs)
            throughputs_df.to_csv('thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '.csv')

            final_throughputs_df = pd.DataFrame(final_throughputs)
            final_throughputs_df.to_csv('thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '_final.csv')

        minthroughputs.append(min(final_throughputs))

        # save at all_throughputs
        all_throughputs += list(final_throughputs)

    # create min throughput information
    minThroughputList.append([iterationCount] + minthroughputs)

    # all_throughputs
    all_throughputs = np.array(all_throughputs)
    all_throughputs = (all_throughputs - np.min(all_throughputs)) / (np.max(all_throughputs) - np.min(all_throughputs))

    # save trajectory as graph
    saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, q, markerColors, training)

# save min throughput as *.csv file
def saveMinThroughput(minThroughputList, memo):

    # save min throughput list as *.csv file
    minThroughputList = pd.DataFrame(np.array(minThroughputList))
    minThroughputList.to_csv('minThroughputList_' + memo + '_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                             '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.csv')

    # save min throughput list as *.txt file
    arr = np.array(minThroughputList)[:, 1:]
    note = 'mean: ' + str(np.mean(arr)) + ', std: ' + str(np.std(arr)) + ', nonzero: ' + str(np.count_nonzero(arr))

    noteFile = open('minThroughputList_' + memo + '_iter_' + ('%04d' % iters) + '_L_' + ('%04d' % L) +
                             '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.txt', 'w')
    
    noteFile.write(note)
    noteFile.close()

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
                                'M':'int', 'L':'int', 'devices':'int', 'T':'float', 'N':'int', 'H':'float',
                                'iters':'int', 'clusteringAtLeast':'float', 'epochs':'int'})

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
    configContent += 'windowSize=' + str(WINDOWSIZE) + '\n'
    configContent += 'clusteringAtLeast=' + str(clusteringAtLeast) + '\n'
    configContent += 'epochs=' + str(epochs)

    configFile.write(configContent)
    configFile.close()

    # minimum throughput list
    minThroughputList = []

    # run training
    try:
        _ = pd.read_csv('input_data_preprocessed.csv' , index_col=0)
        _ = pd.read_csv('output_data_preprocessed.csv', index_col=0)

    except:
        print('[ NO PREPROCESSED DATA ]')

        for iterationCount in range(iters):
            print('TRAINING ITER COUNT ', iterationCount, '/', iters)

            throughputTest(M, T, N, L, devices, width, height, H,
                           ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                           iterationCount, minThroughputList, clusteringAtLeast)

    # get and train model
    try:
        model = tf.keras.models.load_model('WPCN_DL_model')
    except:
        print('model load failed')
        model = getAndTrainModel(epochs)

    # find the best genetic algorithm argument based on the WPCN UAV network environment arguments

    # run test (using 5% (min 10) iterations of training)
    iters = max(10, iters // 20)

    for iterationCount in range(iters):
        print('TEST ITER COUNT ', iterationCount, '/', iters)

        throughputTest(M, T, N, L, devices, width, height, H,
                       ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                       iterationCount, minThroughputList, clusteringAtLeast)
