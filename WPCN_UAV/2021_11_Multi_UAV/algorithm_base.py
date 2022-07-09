# todo:
# input and output data (and how to preprocess) and deep learning model setting
# path finding algorithm (ex: genetic algorithm) and parameters setting

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

# load base settings
baseSettings = h_.loadSettings({'paramCells':'int',
                                'p0_cases':'int',
                                'p1_cases':'int',
                                'p2_cases':'int',
                                'p3_cases':'int',
                                'commOption':'str'}, fileName='base_settings.txt')

base_paramCells  = baseSettings['paramCells']
base_p0_cases    = baseSettings['p0_cases']
base_p1_cases    = baseSettings['p1_cases']
base_p2_cases    = baseSettings['p2_cases']
base_p3_cases    = baseSettings['p3_cases']
base_comm_option = baseSettings['commOption']

# enable GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# allow GPU memory increase
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

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

# the cases are equivalent with below, so REPLACE AS BELOW:
#  x       mod 9 == 0 -> 5m with degree 0, 1 -> 5m with degree 45, ..., 7 -> 5m with degree 315, 8 -> stop

# initial direction: positive direction of x-axis (0 degree)
# q = [[l, t, xlt, ylt, hlt], ...] -> update q
def moveUAV(q, directionList, N, l, width, height):

    for t in range(N-1):

        direction = directionList[t]
        deg = direction

        # check if None value
        if direction == None: continue

        currentLocation = q[l * (N+1) + t][2:] # [xlt, ylt, hlt] of the UAV

        # initialize new X, Y and H
        new_X = currentLocation[0]
        new_Y = currentLocation[1]
        new_H = currentLocation[2]

        # move X, Y with degree 0, 45, ..., 315 or stop
        if direction < 8:
            new_X += 5.0 * math.cos(deg * math.pi / 4)
            new_Y += 5.0 * math.sin(deg * math.pi / 4)

        new_X = np.clip(new_X, 0.0, width)
        new_Y = np.clip(new_Y, 0.0, height)

        # update corresponding q value as new q
        q[l * (N+1) + t+1] = [l, t+1, new_X, new_Y, new_H]

# get the index of the direction
# move UAV using direction info
#  x       mod 9 == 0 -> 5m with degree 0, 1 -> 5m with degree 45, ..., 7 -> 5m with degree 315, 8 -> stop
# (x // 9) mod 3 == 0 -> h+1      , 1 -> h       , 2 -> h-1
def getIndexOfDirection(directionX, directionY):

    # decide x and y change using directionX and directionY
    if abs(directionX) < 2.5 and abs(directionY) < 2.5:
        xy_change = 8
    else:
        arcTan = np.arctan(directionY / directionX) # -90 deg ~  +90 deg
        if arcTan < 0: arcTan += math.pi            #   0 deg ~ +180 deg
        if directionY < 0: arcTan += math.pi        #   0 deg ~ +360 deg

        xy_change = round(arcTan / (math.pi / 4))

    return xy_change

# change color
def changeColor(colorCode, k):
    r = colorCode[1:3]
    g = colorCode[3:5]
    b = colorCode[5:7]

    result = '#'

    for x in [r, g, b]:
        x = int(x, 16)
        
        if x < 127:
            x = max(int(225 * (k - 1)), 0)
        else:
            x = min(int(x * k), 225)

        x = hex(x)

        result += '0'*(2 - len(x[2:])) + x[2:]

    return result

# deep learning model class
class DEEP_LEARNING_MODEL(tf.keras.Model):

    def __init__(self, window, dropout_rate=0.25, training=False, name='WPCN_UAV_model'):
        super(DEEP_LEARNING_MODEL, self).__init__(name=name)
        self.windowSize = window

        # common
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat = tf.keras.layers.Flatten()
        L2 = tf.keras.regularizers.l2(0.001)

        # convolutional neural network part
        self.CNN0 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='valid', activation='relu', name='CNN0') # 18
        self.MaxP0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='MaxPooling0')                # 9
        self.CNN1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu', name='CNN1') # 7
        self.CNN2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', name='CNN2')  # 5

        self.CNNDense0 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=L2, name='CNNDense0')
        self.CNNDense1 = tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=L2, name='CNNDense1')

        # dense part
        self.dense0 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2, name='dense0')
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2, name='dense1')
        self.dense2 = tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=L2, name='dense2')

        # final output part
        self.finalDense = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2, name='dense_final0')
        self.final = tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=L2, name='dense_final1')

    def call(self, inputs, training):
        ws = self.windowSize

        board, parameters = tf.split(inputs, [(2 * ws + 1) * (2 * ws + 1), base_paramCells], axis=1)

        # convolutional neural network part with shape (2 * ws + 1, 2 * ws + 1)
        board = tf.reshape(board, (-1, 2 * ws + 1, 2 * ws + 1, 1))

        board = self.CNN0(board)    # 20 -> 18
        board = self.dropout(board)
        board = self.MaxP0(board)   # 18 -> 9
        board = self.dropout(board)
        board = self.CNN1(board)    # 9  -> 7
        board = self.dropout(board)
        board = self.CNN2(board)    # 7  -> 5

        board = self.flat(board)
        board = self.CNNDense0(board)
        board = self.dropout(board)
        board = self.CNNDense1(board)

        # (skip connection) merge the final dense of the board, with original (board: 4 -> 4 + 4 = 8)
        board = tf.concat([board, parameters], axis=-1)

        # dense part with shape (4)
        params = self.dense0(parameters)
        params = self.dropout(params)
        params = self.dense1(params)
        params = self.dropout(params)
        params = self.dense2(params)

        # final output part
        concatenated = tf.concat([board, params], axis=-1)
        output       = self.finalDense(concatenated)
        output       = self.final(output)

        return output

# define and train model
def defineAndTrainModel(train_input, train_output, test_input, test_output, epochs, windowSize):
    model = DEEP_LEARNING_MODEL(window=windowSize)

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
    model.save('WPCN_UAV_DL_model')

    # test the model (validation)
    test_prediction = model.predict(test_input)
    test_prediction = np.reshape(test_prediction, (len(test_prediction), 1))
    test_result     = np.concatenate((test_prediction, test_output), axis=1)

    test_result     = pd.DataFrame(test_result)
    test_result.to_csv('train_valid_result.csv')

    # return the trained model
    return model

# deep learning model
def getAndTrainModel(epochs, windowSize):

    # load and preprocess input and output data
    try:
        input_data  = pd.read_csv('train_input_preprocessed.csv' , index_col=0)
        output_data = pd.read_csv('train_output_preprocessed.csv', index_col=0)
    except:
        print('[ NO PREPROCESSED DATA ]')
        exit(0)

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
        try:
            with tf.device('/gpu:0'):
                return defineAndTrainModel(train_input, train_output, test_input, test_output, epochs, windowSize)
        except:
            print('GPU load failed -> using CPU')
            with tf.device('/cpu:0'):
                return defineAndTrainModel(train_input, train_output, test_input, test_output, epochs, windowSize)
    except:
        print('Error occurred. If the error is because of the number of COLUMNS in the INPUT data\n' +
              '(for example, 627 = (2*12 + 1) * (2*12 + 1) + 2 when window size is 12 and the number of param is 2)\n' +
              'then remove the model directory (for example, WPCN_UAV_DL_model) and retry.')

# save trajectory as graph

# alkl      : a_l,kl[n] for each UAV l, device k and time slot n
#             where alkl = [[l0, k, l1, n, value], ...

# USING reduced alkl : original   [[l0, k, l1, n, value], ...]
#                      -> reduced [[l0, k, [n0, n1, ...]], ...] or {(l0, k) -> [n0, n1, ...]}
#                      -> (do not update for the device with n_x = 0)

def drawLine(x, y, width, trajectoryArrowThickness, color, isButt):
    if isButt:
        plt.plot(x, y, linewidth=max(0.75, 0.75*width*trajectoryArrowThickness), c=color, solid_capstyle='butt')
    else:
        plt.plot(x, y, linewidth=max(0.75, 0.75*width*trajectoryArrowThickness), c=color)

def saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, all_throughputs_zero,
                        q, markerColors, training, isStatic, L, N,
                        trajectoryArrowLength, trajectoryArrowThickness, alkl):

    plt.clf()
    plt.suptitle('trajectory result at iter ' + str(iterationCount))
    plt.axis([-1, width+1, -1, height+1])

    # w = [[l, k, xkl, ykl, 0], ...] for devices
    for i in range(len(w)):

        # device throughput = 0 (refer to all_throughputs_zero)
        if all_throughputs_zero[i]:
            plt.scatter(w[i][2], w[i][3], s=25,
                        marker='o', c=markerColors[w[i][0]])

        # device throughput > 0
        else:
            plt.scatter(w[i][2], w[i][3],
                        s=int(pow(5.0 + all_throughputs[i] * 7.0, 2.0)),
                        marker='o', c=changeColor(markerColors[w[i][0]], 0.6))

            # communication count (refer to alkl)
            cnt = bin(alkl[str(w[i][0]) + ',' + str(w[i][1])]).count('1')
            plt.text(w[i][2], w[i][3], s=cnt, fontsize=10, c=changeColor(markerColors[w[i][0]], 1.6), weight='bold')

    # q = [[l, t, xlt, ylt, hlt], ...] for UAV
    for l in range(L):
        doNotMoveCnt = 0
        
        for t in range(N):
            ind = l * (N+1) + t

            # draw marker
            if t == 0:
                marker = '+'
                markerSize = 100
            elif t == N-1:
                marker = 'x'
                markerSize = 100
            else:
                marker = 'o'
                markerSize = 25
                
            plt.scatter(q[ind][2], q[ind][3], s=markerSize, marker=marker, c=markerColors[l])

            # draw line
            if t < N-1 and not isStatic:
                x = [q[ind][2],
                     q[ind][2] * trajectoryArrowLength        + q[ind+1][2] * (1.0 - trajectoryArrowLength),
                     q[ind][2] * trajectoryArrowLength * 0.72 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.72),
                     q[ind][2] * trajectoryArrowLength * 0.78 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.78),
                     q[ind][2] * trajectoryArrowLength * 0.47 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.47),
                     q[ind][2] * trajectoryArrowLength * 0.53 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.53),
                     q[ind][2] * trajectoryArrowLength * 0.22 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.22),
                     q[ind][2] * trajectoryArrowLength * 0.28 + q[ind+1][2] * (1.0 - trajectoryArrowLength * 0.28),
                                                                q[ind+1][2]]
                
                y = [q[ind][3],
                     q[ind][3] * trajectoryArrowLength        + q[ind+1][3] * (1.0 - trajectoryArrowLength),
                     q[ind][3] * trajectoryArrowLength * 0.72 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.72),
                     q[ind][3] * trajectoryArrowLength * 0.78 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.78),
                     q[ind][3] * trajectoryArrowLength * 0.47 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.47),
                     q[ind][3] * trajectoryArrowLength * 0.53 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.53),
                     q[ind][3] * trajectoryArrowLength * 0.22 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.22),
                     q[ind][3] * trajectoryArrowLength * 0.28 + q[ind+1][3] * (1.0 - trajectoryArrowLength * 0.28),
                                                                q[ind+1][3]]

                drawLine(x[ :2], y[ :2], 0.0, trajectoryArrowThickness, markerColors[l], False)
                drawLine(x[1:3], y[1:3], 4.0, trajectoryArrowThickness, markerColors[l], True)
                drawLine(x[3:5], y[3:5], 3.0, trajectoryArrowThickness, markerColors[l], True)
                drawLine(x[5:7], y[5:7], 2.0, trajectoryArrowThickness, markerColors[l], True)
                drawLine(x[7: ], y[7: ], 1.0, trajectoryArrowThickness, markerColors[l], True)
            
            else:
                x = [q[ind][2], None]
                y = [q[ind][3], None]

            # check stop of UAV
            if t == N-1 or isStatic or pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2) >= 0.1:
                
                # write "stop count" when starting moving or the last time slot
                if doNotMoveCnt > 0:
                    plt.text(x[0], y[0], s=str(doNotMoveCnt + 1), fontsize=10, weight='bold')
                doNotMoveCnt = 0
                
            else:
                doNotMoveCnt += 1

    # save the figure
    if isStatic:
        plt.savefig('static_trajectory_iter' + ('%04d' % iterationCount))
    elif training == True:
        plt.savefig('train_trajectory_iter' + ('%04d' % iterationCount))
    else:
        plt.savefig('test_trajectory_iter' + ('%04d' % iterationCount))

# find devices within 3m (only x and y)
# w = [[l, k, xkl, ykl, 0], ...]
# q = [[l, t, xlt, ylt, hlt], ...]
def findNearDevice(q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs):

    result = []

    # get (x, y) of the UAV
    ind = l * (N+1) + t
    UAV_x = q[ind][2]
    UAV_y = q[ind][3]

    # find devices within 3m in array 'w'
    for device in w:
        if device[0] == l:
            k = device[1]
            dev_x = device[2]
            dev_y = device[3]

            if pow(UAV_x - dev_x, 2) + pow(UAV_y - dev_y, 2) <= 3 * 3:
                result.append(k)

    # no device within 3m
    if len(result) == 0:
        return algo.findDeviceToCommunicate(q, w, l, t, N, s, b1, b2,
                                            mu1, mu2, fc, c, alphaP, numOfDevs)
    
    # at least 1 devices within 3m
    else:
        return result[t % len(result)]

# update a_l,kl[n]
# alkl      : a_l,kl[n] for each UAV l, device k and time slot n
#             where alkl = [[l0, k, l1, n, value], ...

# USING reduced alkl : original   [[l0, k, l1, n, value], ...]
#                      -> reduced [[l0, k, [n0, n1, ...]], ...] or {(l0, k) -> [n0, n1, ...]}
#                      -> (do not update for the device with n_x = 0)

def update_alkl(alkl, q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs, isStatic):
    devices_in_l = numOfDevs[l]

    # for static mode
    # in sequence -> device 0, 1, ..., (devices-1), 0, 1, ... for each time slot,
    #                until time slot number reaches N
    if isStatic:
        deviceToCommunicate = t % devices_in_l
    else:
        # decide the device to communicate with
        if base_comm_option == 'nearest':
            deviceToCommunicate = algo.findDeviceToCommunicate(q, w, l, t, N, s, b1, b2,
                                                               mu1, mu2, fc, c, alphaP, numOfDevs)

        elif base_comm_option == 'near':
            deviceToCommunicate = findNearDevice(q, w, l, t, N, s, b1, b2,
                                                 mu1, mu2, fc, c, alphaP, numOfDevs)

        else:
            print('base_comm_option of base_settings.txt must be "nearest" or "near".')
            exit(1)

    # when DO NOT have alkl with (l, deviceToCommunicate, l, t)
    if str(l) + ',' + str(deviceToCommunicate) not in alkl:
        alkl[str(l) + ',' + str(deviceToCommunicate)] = pow(2, t)

        # set alkl as 0 for other devices
        for k in range(devices_in_l):
            if k == deviceToCommunicate: continue

            if (str(l) + ',' + str(k) in alkl and
                (alkl[str(l) + ',' + str(k)] % pow(2, t+1)) // pow(2, t) == 1):
                
                alkl[str(l) + ',' + str(k)] = alkl[str(l) + ',' + str(k)] - pow(2, t)
            else:
                pass

    # when have alkl with (l, deviceToCommunicate, l, t)
    else:
        alkl[str(l) + ',' + str(deviceToCommunicate)] += pow(2, t)

        # set alkl as 0 for other devices
        for k in range(devices_in_l):
            if k == deviceToCommunicate: continue

            if (str(l) + ',' + str(k) in alkl and
                (alkl[str(l) + ',' + str(k)] % pow(2, t+1)) // pow(2, t) == 1):
                
                alkl[str(l) + ',' + str(k)] = alkl[str(l) + ',' + str(k)] - pow(2, t)
            else:
                pass

# preprocess input and output
def preprocessInputAndOutput(input_data, output_data, windowSize):
    n = len(input_data)
    assert(n == len(output_data))

    preprocessed_input_data = []
    preprocessed_output_data = []

    # the number of cells representing the input image
    imgCells = (2 * windowSize + 1) * (2 * windowSize + 1)

    for i in range(n):

        # preprocess input data
        preprocessed_input = []

        for j in range(imgCells):
            preprocessed_input.append(input_data[i][j])
        for j in range(base_paramCells):
            preprocessed_input.append(input_data[i][j + imgCells])

        preprocessed_input_data.append(preprocessed_input)

        # preprocess output data (tanh function)
        preprocessed_output_data.append([math.tanh(10 * output_data[i][0])])

    return (preprocessed_input_data, preprocessed_output_data)

# weight w0, w1, w2 and w3 for the base parameter p0, p1, p2 and p3 for the input parameter of the model
def getWeight(base_cases):
    if base_cases == 1:
        return 0
    else:
        return 1 / (base_cases - 1)

# find best parameter for path-finding algorithm
def findBestParams(model, inputImage):

    bestParams = None
    bestOutput = -1.0 # actually at least 0 -> always updated
    outputs    = []

    # weights for p0, p1, p2 and p3
    w0 = getWeight(base_p0_cases)
    w1 = getWeight(base_p1_cases)
    w2 = getWeight(base_p2_cases)
    w3 = getWeight(base_p3_cases)

    for p0 in range(base_p0_cases):
        for p1 in range(base_p1_cases):
            for p2 in range(base_p2_cases):
                for p3 in range(base_p3_cases):
                    params    = [p0 * w0, p1 * w1, p2 * w2, p3 * w3]

                    inputData = np.concatenate((inputImage, params[:base_paramCells]), axis=-1)
                    inputData = np.array([inputData])

                    outputOfModifiedParam = model(inputData, training=False)
                    outputs.append(outputOfModifiedParam[0][0])

                    if outputOfModifiedParam > bestOutput:
                        improvedParams = params
                        bestOutput     = outputOfModifiedParam

                    if improvedParams != None:
                        bestParams = improvedParams
                    else:
                        break

    print('outputs: (for 100 elements with first 100 brute-force params)')
    print(np.round_(np.array(outputs[:100]), 4), '...')
    print('best param: ' + str(bestParams) + ', best output: ' + str(bestOutput))

    return bestParams

# make input image for current UAV location and device location
# dist : cell with ((distance to UAV or device)^2 < sqDist) is marked as -1 (UAV) or 1 (device)
def makeInputImage(q, l, N, w, windowSize, sqDist):

    # current location of UAV [xlt, ylt, hlt]
    UAV_loc = q[l * (N + 1)][2:]
    UAV_x = UAV_loc[0]
    UAV_y = UAV_loc[1]

    inputImage = np.zeros((2 * windowSize + 1, 2 * windowSize + 1))
    dist = int(math.sqrt(sqDist))

    # each device : w = [[l, k, xkl, ykl, 0], ...]
    for device in w:
        if device[0] == l:
            device_x = device[2]
            device_y = device[3]

            relative_device_x = device[2] - UAV_x
            relative_device_y = device[3] - UAV_y

            for dy in range(-dist, dist+1):
                for dx in range(-dist, dist+1):
                    if dy * dy + dx * dx <= sqDist:
                        try:
                            pos_device_x = round(windowSize + relative_device_x) + dx
                            pos_device_y = round(windowSize + relative_device_y) + dy
                            inputImage[pos_device_y][pos_device_x] += (1.0 - (dy * dy + dx * dx) / sqDist)

                            # at most 1.5
                            if inputImage[pos_device_y][pos_device_x] >= 1.5:
                                inputImage[pos_device_y][pos_device_x] = 1.5
                        except:
                            pass

    print(np.round_(inputImage, 2))
    return inputImage

# return the list of device in the current cluster
def findDevicesInCluster(l, deviceList, cluster_mem):
    newDeviceList = []

    for i in range(len(deviceList)):
        if cluster_mem[i] == l:
            newDeviceList.append(deviceList[i])

    return newDeviceList

# running throughput test
def throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU,
                   iterationCount, iters, minThroughputList, clusteringAtLeast, clusteringAtMost,
                   input_data, output_data, training, model, windowSize, isStatic,
                   trajectoryArrowLength, trajectoryArrowThickness,
                   base_func_initializeMovementOfUAV=None,
                   base_func_computeDirectionList=None,
                   base_func_getDeviceLocation=None):

    # create list of devices (randomly place devices)
    while True:
        deviceList = []
        clustering_finished = False

        for i in range(devices):
            xVal = random.random() * width
            yVal = random.random() * height
            device_i = [xVal, yVal]
            deviceList.append(device_i)

        # clustering
        clustering_count = 0

        # create the list of [cluster -> the number of devices] for each cluster and use it
        # for example: 3 devices in cluster 0, 7 devices in cluster 1, and 6 devices in cluster 2
        # then, it becomes [3, 7, 6]
        while True:
            (q, w, cluster_mem, markerColors) = algo.kMeansClustering(L, deviceList, width, height, H, N, False, False, False)
            numOfDevs = [cluster_mem.count(l) for l in range(L)]
            print(numOfDevs)
            clustering_count += 1

            # to prevent stuck in clustering stage
            if clustering_count >= 10:
                print('re-initializing device list ...')
                break

            if min(numOfDevs) >= int(clusteringAtLeast * devices // L) and max(numOfDevs) <= int(clusteringAtMost * devices // L):
                clustering_finished = True
                break

        # clustering finished
        if clustering_finished: break

    # compute common throughput using q and directionList
    # update alkl for each time from 0 to T (N times, N-1 moves)
    alkl = {}

    # al : al[0] for each UAV l
    al = []
    for l in range(L): al.append(algo.transferEnergy(l, l))

    # speed of light
    c = 300000000

    # minimum throughputs
    minthroughputs  = []

    # for trajectory drawing
    all_throughputs = []

    # parameters = using inverse direction of already communicated devices at current
    #              using         direction of remaining            devices at current
    sqDist = 2.0

    for l in range(L):

        # make input image
        inputImage = makeInputImage(q, l, N, w, windowSize, sqDist)
        inputImage = np.array(inputImage).flatten()
        inputImage = np.round_(inputImage, 2)

        # find best parameter using model
        if training == False:
            bestParams = findBestParams(model, inputImage)
            print('\n[ best parameters derived by model ]')

        # decide best parameter randomly
        else:
            bestParams = []
            for i in range(base_paramCells):
                bestParams.append(random.random())
            print('\n[ best parameters derived randomly ]')

        print(np.round_(bestParams, 6))
        print('\n')

        # the list of devices in cluster l
        deviceListC = findDevicesInCluster(l, deviceList, cluster_mem)

        # the number of devices in cluster l
        devices_in_l = numOfDevs[l]
        communicated_devices = []

        if isStatic:
            directionList = [8 for i in range(N)] # UAV does not move
        else:
            directionList = [None for i in range(N)]

        # final throughput (updated for each time t)
        final_throughputs = [0 for i in range(devices_in_l)]

        # initialize movement of UAV between devices
        if base_func_initializeMovementOfUAV != None:
            initialMovement = base_func_initializeMovementOfUAV(devices_in_l)

        # use genetic-like algorithm to decide optimal path :

        # for example,
        # minimum of A*(total movement distance) + B*(sum of 1/d^2 by moving minimum times)
        # parameter 1 -> random swap probability of two neighboring device
        # parameter 2 -> proportion of A and B
        if base_func_computeDirectionList != None and isStatic == False:
            directionList = base_func_computeDirectionList(bestParams, q, l, N, deviceListC, initialMovement, width, height)

        # move UAV from time from 0 to T (N+1 times, N moves), for all UAVs of all clusters
        # (update q)
        if not isStatic:
            moveUAV(q, directionList, N, l, width, height)

        # make direction list using random (when training)
        for t in range(N):

            # update a_l,kl[n] for this (l, t) -> using the UAV location at time t
            update_alkl(alkl, q, w, l, t, N, s, b1, b2, mu1, mu2, fc, c, alphaP, numOfDevs, isStatic)

            # get throughput
            for k in range(devices_in_l):
                thrput = f.formula_11(q, w, l, k, alphaP, N, T, s, b1, b2, mu1, mu2, fc, c, L, alkl, PU, numOfDevs)
                final_throughputs[k] = thrput

            # decide next move (update directionList at time t < N)
            if base_func_getDeviceLocation != None:
                if not isStatic and t < N-1:
                    decision = random.random()
                    currentX = q[l * (N + 1) + t][2]
                    currentY = q[l * (N + 1) + t][3]

                    # decide the direction using the optimal path decided above
                    [goto_X, goto_Y] = base_func_getDeviceLocation(l, w, final_throughputs)

                    directionX = goto_X - currentX
                    directionY = goto_Y - currentY

                    # decide next direction
                    directionList[t] = getIndexOfDirection(directionX, directionY)

        # save throughputs at first iteration
        if iterationCount == 0:
            final_throughputs_df = pd.DataFrame(final_throughputs)

            if isStatic:
                final_throughputs_df.to_csv('static_thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '_final.csv')
            elif training == True:
                final_throughputs_df.to_csv('train_thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '_final.csv')
            else:
                final_throughputs_df.to_csv('test_thrputs_iter_' + str(iterationCount) + '_cluster_' + str(l) + '_final.csv')

        minThrput = min(final_throughputs)
        minthroughputs.append(minThrput)

        # save at all_throughputs
        all_throughputs += list(final_throughputs)

        # add to input and output data
        input_data.append(np.concatenate((inputImage, bestParams), axis=-1))
        output_data.append([minThrput])

    # create min throughput information
    minThroughputList.append([iterationCount] + minthroughputs)

    # save file and trajectory image, not for all iterationCount ...
    if (iterationCount <= 5 or
        (iterationCount <= 50 and iterationCount % 5 == 0) or
        (iterationCount <= 250 and iterationCount % 25 == 0) or
        (iterationCount <= 500 and iterationCount % 100 == 0) or
        iterationCount % 200 == 0 or
        
        (iterationCount == iters // 2 - 1 and isStatic) or                     # static
        (iterationCount == iters - 1      and not isStatic and training) or    # training
        (iterationCount == iters - 1      and not isStatic and not training)): # test

        # save input and output data at the end
        input_data  = np.round_(input_data, 6)
        output_data = np.round_(output_data, 6)
        
        if isStatic:
            pd.DataFrame(np.array(input_data)).to_csv('static_input_raw.csv')
            pd.DataFrame(np.array(output_data)).to_csv('static_output_raw.csv')
        elif training == True:
            pd.DataFrame(np.array(input_data)).to_csv('train_input_raw.csv')
            pd.DataFrame(np.array(output_data)).to_csv('train_output_raw.csv')
        else:
            pd.DataFrame(np.array(input_data)).to_csv('test_input_raw.csv')
            pd.DataFrame(np.array(output_data)).to_csv('test_output_raw.csv')

        # preprocess input and output data
        (preprocessed_input_data, preprocessed_output_data) = preprocessInputAndOutput(input_data,
                                                                                       output_data,
                                                                                       windowSize)

        preprocessed_input_data  = np.round_(preprocessed_input_data, 6)
        preprocessed_output_data = np.round_(preprocessed_output_data, 6)

        if isStatic:
            pd.DataFrame(np.array(preprocessed_input_data)).to_csv('static_input_preprocessed.csv')
            pd.DataFrame(np.array(preprocessed_output_data)).to_csv('static_output_preprocessed.csv')
        elif training == True:
            pd.DataFrame(np.array(preprocessed_input_data)).to_csv('train_input_preprocessed.csv')
            pd.DataFrame(np.array(preprocessed_output_data)).to_csv('train_output_preprocessed.csv')
        else:
            pd.DataFrame(np.array(preprocessed_input_data)).to_csv('test_input_preprocessed.csv')
            pd.DataFrame(np.array(preprocessed_output_data)).to_csv('test_output_preprocessed.csv')

        # all_throughputs
        all_throughputs = np.array(all_throughputs)
        all_throughputs_zero = (all_throughputs == 0)
        all_throughputs = (all_throughputs - np.min(all_throughputs)) / (np.max(all_throughputs) - np.min(all_throughputs))

        # save trajectory as graph
        saveTrajectoryGraph(iterationCount, width, height, w, all_throughputs, all_throughputs_zero,
                            q, markerColors, training, isStatic, L, N,
                            trajectoryArrowLength, trajectoryArrowThickness, alkl)

        # save minimum throughput list
        if isStatic:
            memo = 'static'
        elif training == True:
            memo = 'train'
        else:
            memo = 'test'

        saveMinThroughput(minThroughputList, memo, iters, L, devices, N)

# save min throughput as *.csv file
def saveMinThroughput(minThroughputList, memo, iters, L, devices, N):

    # save min throughput list as *.csv file
    minThroughputList = pd.DataFrame(np.array(minThroughputList))
    minThroughputList.to_csv('minThroughputList_' + memo + '_iter_' + ('%04d' % iters) +
                             '_L_' + ('%04d' % L) + '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.csv')

    # save min throughput list as *.txt file
    arr = np.array(minThroughputList)[:, 1:]
    note = 'mean: ' + str(np.mean(arr)) + ', std: ' + str(np.std(arr)) + ', nonzero: ' + str(np.count_nonzero(arr))

    noteFile = open('minThroughputList_' + memo + '_iter_' + ('%04d' % iters) +
                    '_L_' + ('%04d' % L) + '_devs_' + ('%04d' % devices) + '_N_' + ('%04d' % N) + '.txt', 'w')

    noteFile.write(note)
    noteFile.close()
