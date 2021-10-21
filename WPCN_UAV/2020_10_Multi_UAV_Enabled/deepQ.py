import sys
import math
import copy

import helper as h_
import formula as f
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

import time

# read time check option for getMaxQ(), updateQvalue(), getNextState() and deepLearningQ_test() in deepQ.py
timeCheck = h_.loadSettings({'timeCheck':'logical'})['timeCheck']

# find the maximum value of the number of devices
def getMaxDeviceCount(clusters):
    deviceCount = [len(cluster) for cluster in clusters]
    maxDevices = max(deviceCount)

    return maxDevices

def fillWithConstant(state_array, times):
    const_to_fill = h_.loadSettings({'constantToFill':'float'})['constantToFill']
    
    for i in range(times):
        state_array[1].append(const_to_fill)
        state_array[2].append(const_to_fill)

# q : q[l](t) = (x[l](t), y[l](t), h[l](t))
#     q[n][l] = (x[n][l], y[n][l], h[n][l])

# state: q[n][l], {a[n][l][k_l]}, {R[n][k_l]}
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...
# that is, in the form of UAV[uav_No][x/y/h][time]
def getS(UAV, n, l, a, L, B, PU, g, o2):

    # q[n][l]      (q[n][l])    : the location of UAV l (1d array - x, y and h)
    # a[n][l][k_l] (a[n][l][k]) : the number of times that each device communicates with UAV l (1d array, for all devices)
    # R[n][k_l]    (R[n][l][k]) : the average throughput of devices (for each device k),
    #                             in l-th cluster (1d array, for the devices in l-th cluster)

    # shape of a[n][l] : (K) (within l-th cluster)
    # shape of R       : (K) (within l-th cluster)
    q = f.getqFromUAV(UAV, n)
    devices = len(a[n][l])

    R = []
    for k in range(devices):
        Rnkl = f.R_nkl(L, B, n, l, k, PU, g, o2)
        R.append(Rnkl)

    return [q, a[n][l], R]

# get action with e-greedy while e increases
# with probability e, do the best action
# with probability (1-e), do the action randomly
def getActionWithE(Q, s, e):

    if timeCheck == True: h_.printTime('getActionWithE', 'IN')

    # do action randomly if Q table is empty
    if len(Q) == 0:
        actionIndex = random.randint(0, 3*3*3-1) # (3*3*3) actions in total

        if timeCheck == True: h_.printTime('getActionWithE', 'OUT0')
        return getAction(actionIndex)

    # Q Table           = [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    # convert to input  = converted version of [[s0], [s1], ...]
    #            output = original  version of [[q00, q01, ...], [q10, q11, ...], ...]
    rand = random.random()
    stateIndex = 0 # index of state from the Q Table
    actionIndex = 0 # index of action from the Q Table, from 0 (-1, -1, -1) to 26 (1, 1, 1)

    # find state
    for i in range(len(Q)):
        if s == Q[i][0][0]:
            stateIndex = i
            break

    actions = len(Q[stateIndex][1]) # number of actions (len of [q00, q01, ...])

    # find action from [q00, q01, ...]
    if rand > e: # do the action randomly
        actionIndex = random.randint(0, actions-1)
    else: # do (find) the best action
        actionReward = Q[stateIndex][1][0] # init reward as q00

        # find action from [q00, q01, ...]
        for i in range(actions):
            if Q[stateIndex][1][i] > actionReward:
                actionReward = Q[stateIndex][1][i]
                actionIndex = i

    if timeCheck == True: h_.printTime('getActionWithE', 'OUT1')

    # return action with index of actionIndex
    return getAction(actionIndex)

# get action: from 0 (-1, -1, -1) to 26 (1, 1, 1)
def getAction(actionNo):
    return [int(actionNo / 9) - 1, int((actionNo % 9) / 3) - 1, actionNo % 3 - 1]

# get action space: from (-1, -1, -1) to (1, 1, 1)
def getActionSpace():
    actionSpace = []
    
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                actionSpace.append([x, y, z])

    return actionSpace

# get action index: from (-1, -1, -1) as 0, to (1, 1, 1) as 26
def getActionIndex(action):
    return (action[0]+1)*9 + (action[1]+1)*3 + (action[2]+1)

# get max(a')Q(s', a') (s' = nextState, a' = a_)
# useDL       : whether to use deep learning
# actionSpace : set of actions (in the form of [x, y, z])
# n           : time slot index
# l           : UAV index
# a           : list of a[n][l][k_l] (a part of s)
# R           : list of R[n][k_l]    (a part of s)
def getMaxQ(s, action, n, UAVs, l, a, actionSpace, clusters, B, PU, g, o2, L,
            useDL, trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha):

    if timeCheck == True: h_.printTime('getMaxQ' + ('_DL' if useDL else '_NDL'), 'IN')
    
    # get Q values for the action space of next state s'
    if useDL == True:
        (nextState, _) = getNextState(s, action, n, UAVs, l, a, clusters, B, PU, g, o2, L,
                                      b1, b2, S_, u1, u2, fc, alpha)

        # use R : "corresponding to the UAV index" only
        nextState[2] = nextState[2][l]
        
        # try testing
        try:
            rewardsOfActionsOfNextState = deepLearningQ_test(nextState, False, trainedModel, optimizer, clusters)
            
        except Exception as e:
            useDL = False
            print(e)

    # find optimal action a' = a_ that is corresponding to max(a')Q(s', a')
    # action space = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], ..., [1, 1, 1]]
    maxQ = -999999 # max(a')Q(s', a')

    # when using deep learning, Q value is the maximum value among rewards for actions on the next state
    # otherwise,                Q value is 0
    if useDL == True: maxQ = max(rewardsOfActionsOfNextState[0])
    else: maxQ = 0

    # return
    if timeCheck == True: h_.printTime('getMaxQ' + ('_DL' if useDL else '_NDL'), 'OUT')

    if useDL == True:
        return (maxQ, rewardsOfActionsOfNextState[0])
    else:
        return (maxQ, -1)

# convert state = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}] to "1d array with numeric values"
# q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])

# example of state :
# [[30.710634485123297, 45.731144554946184, 15], ... q[n][l]
#  [1, 1, 1, 1, 1, 0, 2],                        ... {a[n][l]}
#  [0, 0, 0, 0, 0, 0, 0]]                        ... {R[n][l]}
def stateTo1dArray(state):

    q = state[0]
    a = state[1]
    R = state[2]
    
    return q + a + R # because q[n][l], {a[n][l]}, and {R[n][l]} are all 1d arraies

# normalize by each column
def normalize(array, applyLog, title, printAnalysis):

    arr = np.array(copy.deepcopy(array))
    rows = len(arr)
    cols = len(arr[0])

    # x -> log(x+1)
    if applyLog == True:
        for i in range(rows):
            for j in range(cols):
                arr[i][j] = math.log(arr[i][j] + 1, 2)

    # avg and stddev for each column marked as 1, 4 or 5
    avgs = []
    stddevs = []
    
    for j in range(cols):
        col = arr[:, j]
        avgs.append(np.mean(col))
        stddevs.append(np.std(col))

        if printAnalysis == True:

            # save analysis result
            print('\n ==== ' + title + ' col ' + str(j) + ' info ====')
            print('vals =')
            print(str(np.array(col)[:30]))
            print('mean = ' + str(np.mean(col)))
            print('std  = ' + str(np.std(col)))
            print('max  = ' + str(np.max(col)))
            print('min  = ' + str(np.min(col)))

            # save as *.csv file
            fn = 'colAnalysis_' + title + '_' + str(j) + '.csv'
            df = pd.DataFrame(np.array([np.array(col)]).T)
            df.to_csv(fn)

    # convert values
    for i in range(rows):
        for j in range(cols):
            if stddevs[j] == 0: # to fix divide by 0
                arr[i][j] = 0
            else:
                arr[i][j] = (arr[i][j] - avgs[j]) / stddevs[j]

    return arr

# define model (the number of input units not needed)
def defineModel(maxDevices):

    # load settings and find learning rate
    learningRate = h_.loadSettings({'learningRate':'float'})['learningRate']

    # L2 regularization
    L2 = tf.keras.regularizers.l2(0.001)

    # define the model
    model = Sequential()
    model.add(tf.keras.layers.Dense(units=2*maxDevices+3, activation='relu', kernel_regularizer=L2))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=L2))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=L2))
    model.add(tf.keras.layers.Dense(units=27, activation='tanh'))

    adamOpt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

# train data with model
def trainDataWithModel(Q_input, Q_output, model, epochs, iteration, M, episode):

    if timeCheck == True: h_.printTime('trainDataWithModel', 'IN')

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3)
    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_delta=0.0001, mode='min')

    hist = model.fit(Q_input, Q_output, validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)

    model.save('model')

    # save result as *.png image and *.csv file
    # https://snowdeer.github.io/machine-learning/2018/01/11/keras-use-history-function/
    if episode >= 0:

        # *.png file
        plt.clf()
            
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        h_tl = hist.history['loss']
        h_vl = hist.history['val_loss']
        h_ta = hist.history['accuracy']
        h_va = hist.history['val_accuracy']

        loss_ax.plot(h_tl, 'y', label='train loss')
        loss_ax.plot(h_vl, 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')
            
        acc_ax.plot(h_ta, 'b', label='train acc')
        acc_ax.plot(h_va, 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper right')

        plt.title('train result at iter ' + str(iteration) + ' episode ' + str(episode) + ' / ' + str(M))
        plt.savefig('train_result_iter_' + str(iteration) + '_episode_' + str(episode) + '(' + str(len(Q_input)) + ').png')

        # *.csv file
        h_tl = np.array(h_tl)
        h_vl = np.array(h_vl)
        h_ta = np.array(h_ta)
        h_va = np.array(h_va)

        hist_entire = np.column_stack((h_tl, h_vl, h_ta, h_va))
        hist_entire = pd.DataFrame(hist_entire)
        hist_entire.columns = ['train_loss', 'val_loss', 'train_acc', 'val_acc']

        hist_entire.to_csv('train_result_iter_' + str(iteration) + '_episode_' + str(episode) + '.csv')

    if timeCheck == True: h_.printTime('trainDataWithModel', 'OUT')

# deep Learning using Q table (training function)
# printed    : print the detail?
# maxDevices : the maximum number of devices in the cluster
def deepLearningQ_training(Q, deviceName, epoch, printed, iteration, M, episode, clusters):

    h_.printTime('deepLearningQ_training' + ('_DL' if len(Q) > 0 else '_NDL'), 'IN')

    maxDevices = getMaxDeviceCount(clusters)
    model = defineModel(maxDevices)
    epochs = 30
    
    # Q : [state  , action_reward, i (UAV/cluster index), k (device index)]
    #      Q[i][0]  Q[i][1]        Q[i][2]                Q[i][3]
    
    # Q Table           = [[[s0], [Q00, Q01, ...]], [[s1], [Q10, Q11, ...]], ...]
    # convert to input  = converted version of [[s0], [s1], ...]
    #            output = original  version of [[Q00, Q01, ...], [Q10, Q11, ...], ...]
    # where           s = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
    #        and      Q = reward

    # input array (need to convert original array [s0])
    inputData = []
    for i in range(len(Q)):

        # extend {a[n][l][k_l]} and {R[n][k_l]} that the number of items becomes maxDevices
        currentClusterDevices = len(Q[i][0][1])
        fillWithConstant(Q[i][0], maxDevices - currentClusterDevices)
        
        # convert into 1d array (valid if not converted)
        try:
            inputData.append(stateTo1dArray(Q[i][0]))

        # executed if already converted to 1d array
        except:
            inputData.append(Q[i][0])

    # output array (as original)
    outputData = []
    for i in range(len(Q)): outputData.append(Q[i][1])

    # save input and output array as file
    if len(inputData) > 0:
        pd.DataFrame(inputData).to_csv('Q_input.csv')
    if len(outputData) > 0:
        pd.DataFrame(outputData).to_csv('Q_output.csv')

    # save normalized data
    if len(inputData) > 0:
        normalizedInputData = normalize(inputData, False, 'input' + str(len(inputData)), False)
        normalizedOutputData = normalize(outputData, False, 'output' + str(len(inputData)), False)
        pd.DataFrame(normalizedInputData).to_csv('Q_input_normalized.csv')
        pd.DataFrame(normalizedOutputData).to_csv('Q_output_normalized.csv')

    # train using deep learning and save the model (testInputFile and testOutputFile is None)
    # need: modelConfig.txt
    # DON'T NEED TO APPLY SIGMOID to training output data, because DLmain.deeplearning applies it
    try:
        Q_input_noramlized = np.array(pd.read_csv('Q_input_normalized.csv', index_col=0)).astype(float)
        Q_output_noramlized = np.array(pd.read_csv('Q_output_normalized.csv', index_col=0)).astype(float)
        trainDataWithModel(Q_input_noramlized, Q_output_noramlized, model, epochs, iteration, M, episode)

    except:
        print('[train] Q_input_normalized.csv or Q_output_normalized.csv does not exist.')

    h_.printTime('deepLearningQ_training' + ('_DL' if len(Q) > 0 else '_NDL'), 'OUT')

# deep Learning using Q table (test function -> return reward values for each action)
def deepLearningQ_test(state, verbose, trainedModel, optimizer, clusters):

    if timeCheck == True: h_.printTime('deepLearningQ_test', 'IN')

    # convert state into 1d array
    # input unit count = 3 for q + N for {a} + N for {R} = 3 + 2N
    #                    where N is the number of devices in the cluster
    maxDevices = getMaxDeviceCount(clusters)
    currentClusterDevices = len(state[1])
    fillWithConstant(state, maxDevices - currentClusterDevices)
        
    stateArray = [stateTo1dArray(state)]
    stateArray = np.array(stateArray)
    
    # get reward values of the state
    # NEED TO APPLY INV-SIGMOID to test output data, because just getting model output

    # (LARGEST time consumption)
    #testO = trainedModel.predict(stateArray)
    #testO = trainedModel(stateArray, training=False)
    testO = trainedModel(stateArray)
    
    if verbose == True:
        print('test finished')

    if timeCheck == True: h_.printTime('deepLearningQ_test', 'OUT')

    # return output layer
    return testO

# update Q value

# Q Table: [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
# for new state-action-reward-nextState, get reward using deep learning,
# or set to 0 when amount of data is not big enoughly

# Q(s, a) <- (1 - alphaL)*Q(s, a) + alphaL*(Rwd(s, a) + r_*max(a')Q(s', a'))
# because using deep learning, just add Q value for this state and action

# when getting reward, for each UAV,
# 1. the UAV gets direct reward by the situation (in getReward function)
# 2. the state of UAV and the direct reward for the action is stored in the Q table
#    (for all other actions, save 0 as direct reward)
#    2-1. if there is a state that is the same with the state, update Q values for the state

# a            : array {a[n][l][k_l]}
# directReward : direct reward for the action
# alphaL       : learning rate
# r_           : discount factor
# useDL        : TRUE for getting reward using deep learning
#                FALSE for setting to 0
def updateQvalue(Q, QTable, s, action, a, directReward, alphaL, r_, n, UAVs, l, useDL, clusters, B, PU, g, o2, L,
                 trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha):

    if timeCheck == True: h_.printTime('updateQvalue' + ('_DL' if useDL else '_NDL'), 'IN')
    
    # obtain max(a')Q(s', a') (s' = nextState, a' = a_)
    actionSpace = getActionSpace()
    (nextState, deviceToCommunicate) = getNextState(s, action, n, UAVs, l, a, clusters, B, PU, g, o2, L,
                                                    b1, b2, S_, u1, u2, fc, alpha)

    if useDL == True:
        (maxQ, Qvalues) = getMaxQ(s, action, n, UAVs, l, a, actionSpace, clusters, B, PU, g, o2, L,
                                  True, trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha)
        
        # if error for deep Q learning test
        try:
            if Qvalues == -1: useDL = False
        except:
            pass

    else:
        (maxQ, _) = getMaxQ(s, action, n, UAVs, l, a, actionSpace, clusters, B, PU, g, o2, L,
                            False, trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha)

    # update {a[n][l][k_l]} (array of communication times)
    # where k is the index for the device to communicate
    a[n][l] = nextState[1]

    # update Q value when USING DEEP LEARNING
    if useDL == True:
        qs = []

        # i equal to actionIndex -> move to directReward with learning rate alphaL
        # i otherwise            -> Q value
        for i in range(27):
            if i == getActionIndex(action):
                qs.append((1-alphaL) * Qvalues[i] + alphaL * directReward)
            else:
                qs.append(Qvalues[i])

        Q.append([[s], qs, l])
        QTable.append([stateTo1dArray(s), qs, l])

        # update current state s just before return
        s[2] = a[n]

        if timeCheck == True: h_.printTime('updateQvalue' + ('_DL' if useDL else '_NDL'), 'OUT0')

        return

    # update Q value
    # whether the corresponding state is found is NOT IMPORTANT
    # when NOT USING DEEP LEARNING, ALWAYS mark reward element not matching to the ACTION as 0

    # create new action-reward array,
    # and modify the value with corresponding index to the q value for action a
    qs = []
    for i in range(27): qs.append(0)
    actionIndex = getActionIndex(action)
        
    qs[actionIndex] = (1 - alphaL)*qs[actionIndex] + alphaL*(directReward + r_*maxQ)

    # print(str(qs) + ' False ' + str(len(Q)))

    # append the state-action-reward [[s], qs] to Q table
    # Q : [state, action_reward, l (UAV/cluster index)]
    Q.append([[s], qs, l])
    QTable.append([stateTo1dArray(s), qs, l])

    # update current state s
    s[2] = a[n]

    if timeCheck == True: h_.printTime('updateQvalue' + ('_DL' if useDL else '_NDL'), 'OUT1')

# get angle for the location of UAV l q[n][l] and time slot (n-1) to (n)
# q_this   : q[n][l]   (in the [x, y, h] form)
# q_before : q[n-1][l] (in the [x, y, h] form)
# IN RADIAN
def getAngle(q_this, q_before, n):
    
    if n > 0:
        locBefore = q_before
        locAfter = q_this

        # x and y value at time t-1
        xBefore = locBefore[0]
        yBefore = locBefore[1]

        # x and y value at time t
        xAfter = locAfter[0]
        yAfter = locAfter[1]

        # compute angle (angle 0 = positive direction of x axis)
        xDif = xAfter - xBefore
        yDif = yAfter - yBefore

        if yDif == 0 and xDif == 0: return 'not moved' # not moved
        if yDif > 0 and xDif == 0: return math.pi/2 # angle = 90 degree
        elif yDif < 0 and xDif == 0: return 3 * math.pi/2 # angle = 270 degree
        elif yDif >= 0: return math.atan(yDif / xDif) # 0 degree < angle < 180 degree (not 90 degree)
        elif yDif < 0: return math.atan(yDif / xDif) + math.pi # 180 degree < angle < 360 degree (not 270 degree)

    return 'n is not greater than 0'

# get next location when executing action a on state s
# s      : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
# action : action [x, y, z]
# n      : time slot index
# l      : UAV index
# a      : list of a[n][l][k_l] (a part of s)
# R      : list of R[n][k_l]    (a part of s)
def getNextLocation(s, action, n, UAVs, l, a, L, B, PU, g, o2):

    if timeCheck == True: h_.printTime('getNextLocation', 'IN')
    
    # get current location
    curX = s[0][0] # q[n][l][0]
    curY = s[0][1] # q[n][l][1]
    curH = s[0][2] # q[n][l][2]

    # ASSUMPTION:
    # x = -1 : UAV turns to the left (-45 degrees)
    # x = 1  : UAV turns to the right (45 degrees)
    # y = -1 : UAV files backward (-5 m)
    # y = 1  : UAV flies forward (5 m)
    # z = -1 : UAV descends (-1 m)
    # z = 1  : UAV rises (1 m)
    
    # action ([-1, -1, -1] to [1, 1, 1])
    x = action[0]
    y = action[1]
    z = action[2]

    # initialize q_before

    # get angle (radian) between time n-1 and n
    # LOOK AT THE PAPER ONCE MORE TOMORROW
    lastAngle = 0 # set default angle as 0
    for n_ in range(n, 0, -1):

        s_after = getS(UAVs[l], n_, l, a, L, B, PU, g, o2)
        s_before = getS(UAVs[l], n_-1, l, a, L, B, PU, g, o2)
        
        q_after = f.getq(s_after[0][0], s_after[0][1], s_after[0][2])
        q_before = f.getq(s_before[0][0], s_before[0][1], s_before[0][2])
        lastAngle = getAngle(q_after, q_before, n_)
        
        if lastAngle != 'not moved': break # lastAngle is numeric value

    # when all 'not moved' for n, n-1, n-2, ..., 0, then set as default
    if lastAngle == 'not moved': lastAngle = 0

    # get next angle by x
    if x == -1: nextAngle = (lastAngle - math.pi/4) % (math.pi*2)
    elif x == 0: nextAngle = lastAngle
    elif x == 1: nextAngle = (lastAngle + math.pi/4) % (math.pi*2)

    # get next location by x and y
    # initialize as current location
    nextX = curX
    nextY = curY
        
    if y == -1: # backward 5m
        nextX = curX + (-5) * math.cos(nextAngle)
        nextY = curY + (-5) * math.sin(nextAngle)
    elif y == 1: # forward 5m
        nextX = curX + 5 * math.cos(nextAngle)
        nextY = curY + 5 * math.sin(nextAngle)

    # get next height by z
    if z == -1: nextH = curH - 1
    elif z == 0: nextH = curH
    elif z == 1: nextH = curH + 1

    if timeCheck == True: h_.printTime('getNextLocation', 'OUT')

    # return next location
    return [nextX, nextY, nextH]

# get next state
# s       : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
# q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
# action  : action ([-1, -1, -1] to [1, 1, 1])
# l       : UAV index
# a       : list of a[n][l][k_l] (a part of s)
# R       : list of R[n][k_l]    (a part of s)
# L       : the number of clusters = the number of UAVs

## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]

def findDeviceToCommunicate(clusters, n, l, X, Y, H, b1, b2, S_, u1, u2, fc, alpha, g, s):

    actionSpaceSize = len(s[1])
    
    # init next a[n][l]
    next_a = []
    condition = []
    
    # compute g[n][l][k_l] for each device in the cluster
    for k in range(len(clusters[l])):
        PLoS = f.getPLoS(False, l, k, clusters, X, Y, H, b1, b2, S_)
        PNLoS = 1.0 - PLoS
        g[n][l][k][l] = f.g_nlkl(PLoS, u1, PNLoS, u2, fc, l, k, clusters, X, Y, H, alpha)

        condition.append(g[n][l][k][l])
    
    # so, the probability for increasing next_a is 1/len(clusters[l])
    deviceToCommunicate = np.argmax(condition)

    # init to satisfy (5) of the paper
    for i in range(actionSpaceSize): next_a.append(0)
    next_a[deviceToCommunicate] += 1

    # return a of next state
    return (next_a, deviceToCommunicate)
    
def getNextState(s, action, n, UAVs, l, a, clusters, B, PU, g, o2, L,
                 b1, b2, S_, u1, u2, fc, alpha):

    if timeCheck == True: h_.printTime('getNextState', 'IN')
    
    # get next location
    [nextX, nextY, nextH] = getNextLocation(s, action, n, UAVs, l, a, L, B, PU, g, o2)

    # derive next a[n][l]
    # the number of times that each device communicates with UAV l (at time slot n)

    # from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1
    # After flying to the next location,
    # UAV broadcasts energy flow or selects the device that owns the best channel condition
    # in its cluster for uplink communication.

    # find the device with best channel condition (WITHIN CURRENT CLUSTER)
    # note below:
    # n, l, k  : for time, cluster No. and device No.
    # clusters : the location(x and y) of devices for each cluster, shape = (#clusters, #devices_for_each_cluster, 2)
    # nextX    : x axis value of the UAV
    # nextY    : y axis value of the UAV
    # nextH    : h (height), z axis value of the UAV
    
    # additional variables:
    # b1, b2, S_, u1, u2, fc and alpha

    # assumption: select the device to communicate with BEST g[n][l][k_l] value (power gain),
    #             according to page 9126 of the paper

    # find the device to communicate with
    (next_a, deviceToCommunicate) = findDeviceToCommunicate(clusters, n, l, nextX, nextY, nextH, b1, b2, S_, u1, u2, fc, alpha, g, s)

    # derive next R[n][k_l]
    # the average throughput of devices in l-th cluster
    
    nextR = []
    for UAV in range(len(clusters)):
        
        nextR_UAV = []
        for k in range(len(clusters[UAV])): # for each device in cluster l
            nextR_UAV.append(f.R_nkl(L, B, n, UAV, k, PU, g, o2))
        nextR.append(nextR_UAV)

    #### return ####
    # s' = [q'[n][l], {a'[n][l][k_l]}, {R'[n][k_l]}]
    if timeCheck == True: h_.printTime('getNextState', 'OUT')

    return ([[nextX, nextY, nextH], next_a, nextR], deviceToCommunicate)

# target Q value yt = r + r_*max(a')Q(s', a', w) = r + r_*max(a')Q(s', a')
# useDL : whether to use deep learning
def yt(r, r_, s, action, n, UAVs, l, a, actionSpace, clusters, B, PU, g, l_, o2, L,
       useDL, trainedModel, optimizer):
    
    (maxQ, _) = getMaxQ(s, action, n, UAVs, l, a, actionSpace, clusters, B, PU, g, l_, o2, L,
                        useDL, trainedModel, optimizer, b1, b2, S_, u1, u2, fc, alpha)
    return r + r_ * maxQ
    
# Q^pi(s, a) = E[Sum(k=0, inf)(r_^k * r_(t+k)) | st, at, pi]
#def Qpi(r_, r, t, k, s, a, pi):
#    # ( [6] get Q(s, a) using policy pi )
