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
# q = [[l, t, xlt, ylt, hlt], ...] -> update q
def moveUAV(q, directionList, N, L, width, height):

    for l in range(L):
        degree = 0.0
        
        for t in range(N):

            direction = directionList[l * N + t]

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

# make training dataset
# input       : device location (w[:, 2], w[:, 3]) + (height of UAV) + action (dif of x, y, and h of UAV)
# output      : function of throughput change
# action_list : directionList[l*N : l*(N + 1)]

# cluster No. : w[:, 0]
# throughputs : throughputs at time slot t = 0 ... N-1
def modifyArr(arr, y, x, value, window):
    try:
        arr[y + window][x + window] += value

        # clip
        if arr[y + window][x + window] < -1.5:
            arr[y + window][x + window] = -1.5
        elif arr[y + window][x + window] > 1.5:
            arr[y + window][x + window] = 1.5
    except:
        pass

# mark the position of device
def markDevicePosition(board, board_x, board_y, thrput, window):

    distrib = [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.05, 1.1,
               1.05, 1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.05]

    for y in range(19):
        for x in range(19):
            yy = y + (board_y - 9)
            xx = x + (board_x - 9)
            
            modifyArr(board, yy, xx, distrib[y] * distrib[x] * thrput, window)

# get device positions
def getDevicePos(w, l):

    # filter for cluster l
    dev_x = []
    dev_y = []
    
    for i in range(len(w)):
        if w[i][0] == l:
            dev_x.append(w[i][2])
            dev_y.append(w[i][3])

    return (dev_x, dev_y)

# get distance between the device (with index) and UAV
def getDistBetweenDeviceAndUAV(i, dev_x, dev_y, UAV_x, UAV_y):
    return pow(pow(dev_x[i] - UAV_x, 2.0) + pow(dev_y[i] - UAV_y, 2.0), 0.5)

# thrput: common throughput value at time t
def makeBoard(thrput, w, l, window, width, height):

    board = np.zeros((width + 2 * window, height + 2 * window))

    # max device and min device
    maxThrput = max(thrput)
    minThrput = min(thrput)

    # get device positions
    (dev_x, dev_y) = getDevicePos(w, l)

    # the number of devices in cluster l
    dev_in_l = len(dev_x)

    # normalized throughput
    if len(thrput) >= 2:
        thrput_ = [(thrput[i] - minThrput) / (maxThrput - minThrput) * 2.0 - 1.0 for i in range(len(thrput))]
    else:
        thrput_ = np.array([0])

    for i in range(dev_in_l):

        # place the device on the board (mark as [[     0.2, 0.5, 0.2     ],
        #                                         [0.2, 0.6, 1.0, 0.6, 0.2],
        #                                         [0.5, 1.0, 1.2, 1.0, 0.5],
        #                                         [0.2, 0.6, 1.0, 0.6, 0.2],
        #                                         [     0.2, 0.5, 0.2     ]])
        board_x = int(dev_x[i])
        board_y = int(dev_y[i])

        # mark the position of device
        markDevicePosition(board, board_x, board_y, thrput_[i], window)

    return board

# compute output (reward)
# q_current, q_after   : the location of UAV
# thrput, thrput_after : throughput values of device
# w                    : device location info
def computeOutput(q_current, q_after, thrput, thrput_after, w, l):

    """
    print('q_current   :', np.shape(q_current))
    print(np.array(q_current))
    print('\nq_after     :', np.shape(q_after))
    print(np.array(q_after))
    print('\nthrput      :', np.shape(thrput))
    print(np.array(thrput))
    print('\nthrput_after:', np.shape(thrput_after))
    print(np.array(thrput_after))
    print('\nw           :', np.shape(w))
    print(np.round_(w, 4))
    print('\nl           :', l)
    exit(0)
    """

    try:
        after  = np.mean(thrput_after) / np.max(thrput_after)
        before = np.mean(thrput)       / np.max(thrput)
        output = 1.0 / (1.0 + math.exp(-(after - before) * 400))
    except:
        output = 0.5

    return output

# make input and output based on current HAP location
def makeInputAndOutput(q_current, q_after, thrput, thrput_after, board, window,
                       iterationCount, w, l, t):

    # get device positions
    (dev_x, dev_y) = getDevicePos(w, l)

    #### INPUT
    # board (center: x and y of UAV)
    center_x = int(q_current[0])
    center_y = int(q_current[1])
    
    input_board = board[center_y : center_y + 2 * window,
                        center_x : center_x + 2 * window]

    # mark the center
    """
    for y in range(-2, 2):
        for x in range(-2, 2):
            input_board[window + x][window + y] = 1.5
    """

    # height of UAV
    UAVheight = np.array([q_current[2]])

    # action (dif of x, y, and h of UAV)
    action = np.array(q_after) - np.array(q_current)

    #### OUTPUT
    # compute output (reward) based on throughput change
    output = computeOutput(q_current, q_after, thrput, thrput_after, w, l)

    # plot the board array using seaborn
    if iterationCount == 0 and l < 2 and t < 5:
        plt.clf()
        ax = sns.heatmap(input_board)
        plt.savefig('input_board_train_' + str(iterationCount) + ',' + str(l) + ',' + str(t) + '.png',
                    bbox_inches='tight', dpi=100)

        #plt.clf()
        #ax = sns.heatmap(board)
        #plt.savefig('input_board_' + str(iterationCount) + ',' + str(l) + ',' + str(t) + '_original.png',
        #            bbox_inches='tight', dpi=100)

    # save training dataset
    # input  : board + height + action
    # output : output
    input_  = np.concatenate((input_board.flatten(), UAVheight, action), -1)
    output_ = np.array([output])

    return (input_, output_)

# make training dataset
def makeTrainDataset(w, l, throughputs, t, q, iterationCount):

    w = np.array(w)
    
    # board configuration
    width  = h_.loadSettings({'width':'int'})['width']
    height = h_.loadSettings({'height':'int'})['height']

    thrput       = throughputs[t]
    thrput_after = throughputs[t+1]

    q_current = q[l * (N+1) + t    ][2:5]
    q_after   = q[l * (N+1) + (t+1)][2:5]

    # find the range of the board (unit: width=1.0m, height=1.0m)
    window = WINDOWSIZE

    # make the board for training
    board = makeBoard(thrput, w, l, window, width, height)

    # make input data based on current HAP location
    (input_, output_) = makeInputAndOutput(q_current, q_after, thrput, thrput_after, board, window,
                                           iterationCount, w, l, t)

    return (input_, output_)

# make test input (test input data) based on current HAP location
# with action (dif of x, y, and h of UAV)
def makeInputForTest(q_current, thrput, board, window, w, l, t, action, iterationCount):

    # get device positions
    #print('A', time.time(), len(w))
    (dev_x, dev_y) = getDevicePos(w, l)

    #### INPUT
    # board (center: x and y of UAV)
    #print('B', time.time())
    center_x = int(q_current[0])
    center_y = int(q_current[1])
    
    input_board = board[center_y : center_y + 2 * window,
                        center_x : center_x + 2 * window]

    # height of UAV
    #print('C', time.time())
    UAVheight = np.array([q_current[2]])
    
    # plot the board array using seaborn
    #print('D', time.time(), iterationCount, l, t)
    if iterationCount == 0 and l < 2 and t < 5:
        plt.clf()
        ax = sns.heatmap(input_board)
        plt.savefig('input_board_test_' + str(iterationCount) + ',' + str(l) + ',' + str(t) + '.png',
                    bbox_inches='tight', dpi=100)
    
    # save training dataset
    # input  : board + height + action
    # output : output
    #print('E', time.time())
    input_  = np.concatenate((input_board.flatten(), UAVheight, action), -1)
    #print('F', time.time())
    return input_

# make test input data
def makeTestInput(w, l, throughputs, t, q, action, width, height, iterationCount):

    #print('a', time.time())
    w = np.array(w)
    
    # board configuration
    #print('b', time.time())
    thrput    = throughputs[t]
    #print('c', time.time())
    q_current = q[l * (N+1) + t][2:5]
    
    # find the range of the board (unit: width=1.0m, height=1.0m)
    #print('d', time.time())
    window = WINDOWSIZE

    # make the board for training
    #print('e', time.time())
    board = makeBoard(thrput, w, l, window, width, height)

    # make input data based on current HAP location
    #print('f', time.time())
    input_ = makeInputForTest(q_current, thrput, board, window, w, l, t, action, iterationCount)

    return input_

# deep learning model class
class DEEP_LEARNING_MODEL(tf.keras.Model):

    def __init__(self, window=WINDOWSIZE, dropout_rate=0.25, training=False, name='WPCN_model'):

        super(DEEP_LEARNING_MODEL, self).__init__(name=name)
        self.windowSize = window

        # common
        self.dropout  = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat     = tf.keras.layers.Flatten()
        L2            = tf.keras.regularizers.l2(0.001)

        # CNN layers for board (2 * window) * (2 * window)
        self.CNN0     = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='valid', activation='relu', name='CNN0')
        self.MaxP0    = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='MaxPooling0')
        self.CNN1     = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu', name='CNN1')
        self.MaxP1    = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='MaxPooling1')
        self.CNN2     = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', name='CNN2')
        #self.MaxP1    = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', name='MaxPooling1')
        #self.CNN3     = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', name='CNN3')
        #self.CNN4     = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation='relu', name='CNN4')

        self.den_CNN0 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_CNN0')
        self.den_CNN1 = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=L2, name='dense_CNN1')
        
        # final
        self.final    = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        ws = self.windowSize

        # split inputs:   board         (2 * window)*(2 * window)
        #               + height of UAV (1)
        #               + action        (3)
        board, UAV_height, actionInfo = tf.split(inputs, [(2 * ws) * (2 * ws), 1, 3], axis=1)

        # CNN layers for board (2 * window)*(2 * window)
        board = tf.reshape(board, (-1, (2 * ws), (2 * ws), 1))

        board = self.CNN0(board)    # 30 -> 28
        board = self.dropout(board)
        board = self.MaxP0(board)   # 28 -> 14
        
        board = self.CNN1(board)    # 14 -> 12
        board = self.dropout(board)
        board = self.MaxP1(board)   # 12 -> 6
        board = self.dropout(board)
        board = self.CNN2(board)    # 6 -> 4
        
        board = self.flat(board)
        board = self.den_CNN0(board)
        board = self.den_CNN1(board)

        # final
        concatenated = tf.concat([board, UAV_height, actionInfo], axis=-1)
        output = self.final(concatenated)

        return output

# preprocess input data
def preprocessInput(input_data, BOARD_ARGS):

    # board (2*window)*(2*window) -> x := x / 1.5
    input_data[:, :BOARD_ARGS] = input_data[:, :BOARD_ARGS] / 1.5
    
    # height of UAV (1) -> x := N(x|Mu=0, Sigma=1)
    # suppose that original Mu=15.0 and Sigma=2.5
    input_data[:, BOARD_ARGS] = (input_data[:, BOARD_ARGS] - 15.0) / 2.5
    
    # action info (3) -> x := x / 5.0 for x and y
    #                    x := x       for h
    input_data[:, BOARD_ARGS+1:BOARD_ARGS+3] = input_data[:, BOARD_ARGS+1:BOARD_ARGS+3] / 5.0
    
# preprocess input and output data    
def preprocessData():

    BOARD_ARGS = 4 * WINDOWSIZE * WINDOWSIZE # (2 * WINDOWSIZE)^2

    # load original input and output data
    new_input_data  = pd.read_csv('input_data.csv' , index_col=0)
    new_output_data = pd.read_csv('output_data.csv', index_col=0)

    new_input_data  = np.array(new_input_data)
    new_output_data = np.array(new_output_data)

    # preprocess input data
    preprocessInput(new_input_data, BOARD_ARGS)

    # preprocess output data
    # DO NOTHING

    # remove digits to reduce volume of the *.csv files
    new_input_data  = np.round_(new_input_data , 4)
    new_output_data = np.round_(new_output_data, 6)

    # save preprocessed data
    new_input_data_df  = pd.DataFrame(new_input_data)
    new_output_data_df = pd.DataFrame(new_output_data)

    new_input_data_df.to_csv('input_data_preprocessed.csv')
    new_output_data_df.to_csv('output_data_preprocessed.csv')

    # return preprocessed data
    return (new_input_data, new_output_data)

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

# move the UAV using learned model
def moveUAV_DL(board, UAVheight, q, model, w, l, N, t, throughputs, iterationCount):

    # make input data using (board + height of UAV + action)
    # get output data of the model for each action (board + height of UAV + action)
    # (in this way, CONSEQUENTLY SAME with the original movement set)
    actions = []
    dif_x         = [math.sqrt(12.5), 0, -math.sqrt(12.5), -5, -math.sqrt(12.5), 0, math.sqrt(12.5), 5, 0]
    dif_y         = [math.sqrt(12.5), 5, math.sqrt(12.5), 0, -math.sqrt(12.5), -5, -math.sqrt(12.5), 0, 0]
    height_change = [0, 1, -1]
    
    for i in range(3 * 3):
        for j in range(3):
            actions.append([dif_x[i], dif_y[i], height_change[j]])

    # find the best action with maximum output value
    outputs = []

    print('making test input:', l, t)

    width  = h_.loadSettings({'width':'int'})['width']
    height = h_.loadSettings({'height':'int'})['height']
    
    for action in actions:
        #print(1, time.time())
        input_  = makeTestInput(w, l, throughputs, t, q, action, width, height, iterationCount)
        #print(2, time.time())
        input_  = np.array([input_])
        #print(3, time.time())
        preprocessInput(input_, 4 * WINDOWSIZE * WINDOWSIZE) # preprocess input first
        #print(4, time.time())
        output_ = model(input_, training=False) # model.predict(input_)
        #print(5, time.time())
        outputs.append(output_[0][0])

    # index of the best action
    bestAction = np.argmax(outputs)

    # find the UAV to move, using l and t
    # move UAV using the best action (update q)
    new_X = q[l * (N+1) + t][2] + actions[bestAction][0]
    new_Y = q[l * (N+1) + t][3] + actions[bestAction][1]
    new_H = q[l * (N+1) + t][4] + actions[bestAction][2]

    # adjust new_X, new_Y and new_H using width and height
    new_X = np.clip(new_X, 0.0, width)
    new_Y = np.clip(new_Y, 0.0, height)
    if new_H < 0.0: new_H = 0.0
    
    q[l * (N+1) + t+1] = [l, t+1, new_X, new_Y, new_H]

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

def throughputTest(M, T, N, L, devices, width, height, H,
                   ng, fc, B, o2, b1, b2, alphaP, alphaL, mu1, mu2, s, PD, PU,
                   iterationCount, minThroughputList, clusteringAtLeast,
                   input_data, output_data, training, model):

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

        # throughput results
        throughputs = []

        # the number of devices in cluster l
        devices = numOfDevs[l]
        directionList = [None for i in range(L * N)]
        
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

                #for i in range(len(throughputs)):
                #    print(i, np.round_(throughputs[i], 3)) 

            throughputs.append(thrputs)

            if training == True:

                # make direction list using random (when training)
                directionList[l * N + t] = random.randint(0, 26)

                # move UAV from time from 0 to T (N+1 times, N moves), for all UAVs of all clusters
                # (update q)
                moveUAV(q, directionList, N, L, width, height)

            # [ TEST ]
            # move UAV based on the model
            else:

                # assert that model exists
                assert(model != None)
                
                # create board
                # np.shape of thrput should be (number_of_devices,)
                board = makeBoard(thrputs, w, l, WINDOWSIZE, int(width), int(height))

                # find height of UAV
                UAVheight = q[l * (N+1) + t][4]

                # move the UAV using deep learning result (update q)
                moveUAV_DL(board, UAVheight, q, model, w, l, N, t, throughputs, iterationCount)

        # [ TRAINING ]
        # make training dataset
        if training == True:
            for t in range(N-1):
                
                (input_, output_) = makeTrainDataset(w, l, throughputs, t, q, iterationCount)
                
                input_data .append(list(input_ ))
                output_data.append(list(output_))

        # throughputs: shape (k, N) -> shape (N, k)
        throughputs       = np.array(throughputs)
        final_throughputs = throughputs[-1]

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

# main TRAINING code
def train(iters, M, T, N, L, devices, width, height, H,
          ng, fc, B, o2, b1, b2, alphaP, mu1, mu2, s, PU,
          clusteringAtLeast):

    # list of min throughputs
    minThroughputList = []

    # input and output data
    input_data  = []
    output_data = []

    # main training
    for iterationCount in range(iters):
        print('ITER COUNT ', iterationCount, '/', iters)
        
        throughputTest(M, T, N, L, devices, width, height, H,
                       ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                       iterationCount, minThroughputList, clusteringAtLeast,
                       input_data, output_data, training=True, model=None)

    # save input and output data
    input_data  = pd.DataFrame(input_data)
    output_data = pd.DataFrame(output_data)

    # remove digits to reduce volume of the *.csv files
    input_data  = np.round_(input_data , 4)
    output_data = np.round_(output_data, 6)

    input_data .to_csv('input_data.csv')
    output_data.to_csv('output_data.csv')

    # save min throughput as *.csv file
    saveMinThroughput(minThroughputList, 'train')

# main TEST code
def test(iters, M, T, N, L, devices, width, height, H,
         ng, fc, B, o2, b1, b2, alphaP, mu1, mu2, s, PU,
         clusteringAtLeast, model):

    # re-initialize minimum throughput list
    minThroughputList = []

    # main test
    for iterationCount in range(iters):
        print('ITER COUNT ', iterationCount)
        
        throughputTest(M, T, N, L, devices, width, height, H,
                       ng, fc, B, o2, b1, b2, alphaP, None, mu1, mu2, s, None, PU,
                       iterationCount, minThroughputList, clusteringAtLeast,
                       input_data=None, output_data=None, training=False, model=model)

    # save min throughput as *.csv file
    saveMinThroughput(minThroughputList, 'test')

if __name__ == '__main__':

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

    # run training
    try:
        _ = pd.read_csv('input_data_preprocessed.csv' , index_col=0)
        _ = pd.read_csv('output_data_preprocessed.csv', index_col=0)

    except:
        print('[ NO PREPROCESSED DATA ]')
        train(iters, M, T, N, L, devices, width, height, H,
              ng, fc, B, o2, b1, b2, alphaP, mu1, mu2, s, PU,
              clusteringAtLeast)

    # get and train model
    try:
        model = tf.keras.models.load_model('WPCN_DL_model')
    except:
        print('model load failed')
        model = getAndTrainModel(epochs)

    # run test (using 5% (min 10) iterations of training)
    iters = max(10, iters // 20)
    test(iters, M, T, N, L, devices, width, height, H,
         ng, fc, B, o2, b1, b2, alphaP, mu1, mu2, s, PU,
         clusteringAtLeast, model)
