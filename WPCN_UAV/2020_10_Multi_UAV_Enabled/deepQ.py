import sys
import math
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main
import formula as f
import random
import readData as RD
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

# q : q[l](t) = (x[l](t), y[l](t), h[l](t))
#     q[n][l] = (x[n][l], y[n][l], h[n][l])

# state: q[n][l], {a[n][l][k_l]}, {R[n][k_l]}
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...
# that is, in the form of UAV[uav_No][x/y/h][time]
def getS(UAV, n, l, a, R):

    # q[n][l]      (q[n][l]) : the location of UAV l (1d array - x, y and h)
    # a[n][l][k_l] (a[n][l]) : the number of times that each device communicates with UAV l (1d array, for all devices)
    # R[n][k_l]    (R[n])    : the average throughput of devices (for each device k),
    #                          in l-th cluster (1d array, for the devices in l-th cluster)
    q = f.getqFromUAV(UAV, n)
    
    return [q, a[n][l], R[n]]

# get action with e-greedy while e increases
# with probability e, do the best action
# with probability (1-e), do the action randomly
def getActionWithE(Q, s, e):

    # do action randomly if Q table is empty
    if len(Q) == 0:
        actionIndex = random.randint(0, 3*3*3-1) # (3*3*3) actions in total
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
def getMaxQ(s, action, n, UAVs, l, k, a, R, actionSpace, clusters, B, PU, g, l_, o2, useDL):

    # get Q values for the action space of next state s'
    if useDL == True:
        (nextState, _) = getNextState(s, action, n, UAVs, l, a, R, clusters, B, PU, g, l_, o2)
        
        # try testing
        try:
            rewardsOfActionsOfNextState = deepLearningQ_test(nextState, k, False)
        except Exception as e:
            useDL = False
            print(' === error message ===')
            print(e)

    # find optimal action a' = a_ that is corresponding to max(a')Q(s', a')
    # action space = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], ..., [1, 1, 1]]
    maxQ = -999999 # max(a')Q(s', a')

    # when using deep learning, Q value is the maximum value among rewards for actions on the next state
    # otherwise,                Q value is 0
    if useDL == True: maxQ = max(rewardsOfActionsOfNextState[0])
    else: maxQ = 0

    # return
    if useDL == True:
        return (maxQ, rewardsOfActionsOfNextState[0])
    else:
        return (maxQ, -1)

# convert state = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}] to "1d array with numeric values"
# q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])

# example of state :
# [[30.710634485123297, 45.731144554946184, 15],              ... q[n][l]
#  [1, 1, 1, 1, 1, 0, 2],                                     ... a[n][l][k_l]
#  [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0],     ... {R[n][k_l]}
#   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
def stateTo1dArray(state, k):

    q = state[0]
    a = state[1][k]

    R = []
    for i in range(len(state[2])):
        R += state[2][i]
    
    return q + [a] + R # because q[n][l], a[n][l], and R[n][l] are all 1d arraies

# deep Learning using Q table (training function)
# printed : print the detail?
def deepLearningQ_training(Q, deviceName, epoch, printed):
    
    # Q : [state, action_reward, i (UAV/cluster index), k (device index)]
    
    # Q Table           = [[[s0], [Q00, Q01, ...]], [[s1], [Q10, Q11, ...]], ...]
    # convert to input  = converted version of [[s0], [s1], ...]
    #            output = original  version of [[Q00, Q01, ...], [Q10, Q11, ...], ...]
    # where           s = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
    #        and      Q = reward

    # input array (need to convert original array [s0])
    inputData = []
    for i in range(len(Q)):
        inputData.append(stateTo1dArray(Q[i][0], Q[i][3]))

    # output array (as original)
    outputData = []
    for i in range(len(Q)): outputData.append(Q[i][1])

    # save input and output array as file
    if len(inputData) > 0:
        RD.saveArray('Q_input.txt', inputData)
    if len(outputData) > 0:
        RD.saveArray('Q_output.txt', outputData)

    # train using deep learning and save the model (testInputFile and testOutputFile is None)
    # need: modelConfig.txt
    # DON'T NEED TO APPLY SIGMOID to training output data, because DLmain.deeplearning applies it
    try:
        DLmain.deepLearning('Q_input.txt', 'Q_output.txt', None, None, None,
                            None, 0.0, None, 'modelConfig.txt', deviceName, epoch, printed, 'deepQ_model')
    except:
        print('Q_input.txt or Q_output.txt does not exist.')

# deep Learning using Q table (validation)
def deepLearningQ_valid(deviceName, epoch, printed, validRate):
    
    try:
        DLmain.deepLearning('Q_input.txt', 'Q_output.txt', None, None, None,
                            None, validRate, 'Q_valid_report.txt', 'modelConfig.txt', deviceName, epoch, printed, 'deepQ_model')
    except:
        print('Q_input.txt or Q_output.txt does not exist.')

# deep Learning using Q table (test function -> return reward values for each action)
def deepLearningQ_test(state, k, verbose):

    # convert state into 1d array
    stateArray = stateTo1dArray(state, k)

    # get reward values of the state
    # NEED TO APPLY INV-SIGMOID to test output data, because just getting model output
    optimizer = tf.keras.optimizers.Adam(0.001)
    trainedModel = DL.deepLearningModel('deepQ_model', optimizer, 'mse', verbose)

    testO = DL.modelOutput(trainedModel, [stateArray])
    outputLayer = testO[len(testO)-1]

    # apply inverse sigmoid to output values
    for i in range(len(outputLayer)): # for each output data
        for j in range(len(outputLayer[0])): # for each value of output data
            outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])

    if verbose == True: print('test finished')

    # return output layer
    return outputLayer

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
def updateQvalue(Q, s, action, a, directReward, alphaL, r_, n, UAVs, l, k, R, useDL, clusters, B, PU, g, l_, o2):

    # obtain max(a')Q(s', a') (s' = nextState, a' = a_)
    actionSpace = getActionSpace()
    (nextState, deviceToCommunicate) = getNextState(s, action, n, UAVs, l, a, R, clusters, B, PU, g, l_, o2)

    if useDL == True:
        (maxQ, Qvalues) = getMaxQ(s, action, n, UAVs, l, k, a, R, actionSpace, clusters, B, PU, g, l_, o2, True)

        # if error for deep Q learning test
        try:
            if Qvalues == -1: useDL = False
        except:
            pass
    else:
        (maxQ, _) = getMaxQ(s, action, n, UAVs, l, k, a, R, actionSpace, clusters, B, PU, g, l_, o2, False)

    # update {a[n][l][k_l]} (array of communication times)
    # where k is the index for the device to communicate
    a[n][l] = nextState[1]

    # update Q value when using deep learning
    if useDL == True:
        qs = []

        # i equal to actionIndex -> move to directReward with learning rate alphaL
        # i otherwise            -> Q value
        for i in range(27):
            if i == getActionIndex(action):
                qs.append((1-alphaL) * Qvalues[i] + alphaL * directReward)
            else:
                qs.append(Qvalues[i])

        Q.append([[s], qs, l, k])
        print(len(Q))

        return
        
    # update Q value
    sFound = False # find corresponding state?
    
    for i in range(len(Q)):

        if Q[i][0][0] == s: # Q[i] = [[s0], [q00, q01, ...]], Q[i][0] = [s0], Q[i][0][0] = s0

            # print(str(Q[i][1]) + ' True ' + str(len(Q)))
            
            actionIndex = getActionIndex(action)
            Q[i][1][actionIndex] = (1 - alphaL)*Q[i][1][actionIndex] + alphaL*(directReward + r_*maxQ)

            sFound = True
            break

    # when corresponding state is not found
    if sFound == False:

        # create new action-reward array,
        # and modify the value with corresponding index to the q value for action a
        qs = []
        for i in range(27): qs.append(0)
        actionIndex = getActionIndex(action)
        
        qs[actionIndex] = (1 - alphaL)*qs[actionIndex] + alphaL*(directReward + r_*maxQ)

        # print(str(qs) + ' False ' + str(len(Q)))

        # append the state-action-reward [[s], qs] to Q table
        # Q : [state, action_reward, l (UAV/cluster index), k (device index)]
        Q.append([[s], qs, l, k])

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
def getNextLocation(s, action, n, UAVs, l, a, R):
    
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

        s_after = getS(UAVs[l], n_, l, a, R)
        s_before = getS(UAVs[l], n_-1, l, a, R)
        
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

    # return next location
    return [nextX, nextY, nextH]

# get next state
# s       : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
# q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
# action  : action ([-1, -1, -1] to [1, 1, 1])
# l       : UAV index
# a       : list of a[n][l][k_l] (a part of s)
# R       : list of R[n][k_l]    (a part of s)

## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]
def getNextState(s, action, n, UAVs, l, a, R, clusters, B, PU, g, l_, o2):

    # get next location
    [nextX, nextY, nextH] = getNextLocation(s, action, n, UAVs, l, a, R)

    # derive next a[n][l]
    # the number of times that each device communicates with UAV l (at time slot n)

    # from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1
    # After flying to the next location,
    # UAV broadcasts energy flow or selects the device that owns the best channel condition
    # in its cluster for uplink communication.

    # find the device with best channel condition
    # NO NEED TO WRITE CODE FOR IT because RANDOMLY SELECT the device

    # init next a[n][l] as current a[n][l] from s (a number, not an array)
    next_a = s[1]
    
    # assumption: randomly select the device to communicate with
    # so, the probability for increasing next_a is 1/len(clusters[l])
    deviceToCommunicate = random.randint(0, len(clusters[l])-1)
    next_a[deviceToCommunicate] += 1

    # derive next R[n][k_l]
    # the average throughput of devices in l-th cluster
    
    nextR = []
    for UAV in range(len(clusters)):
        
        nextR_UAV = []
        for k in range(len(clusters[UAV])): # for each device in cluster l
            nextR_UAV.append(f.R_nkl(B, k, UAV, n, PU, g, l_, o2))
        nextR.append(nextR_UAV)

    #### return ####
    # s' = [q'[n][l], {a'[n][l][k_l]}, {R'[n][k_l]}]
    return ([[nextX, nextY, nextH], next_a, nextR], deviceToCommunicate)

# target Q value yt = r + r_*max(a')Q(s', a', w) = r + r_*max(a')Q(s', a')
# useDL : whether to use deep learning
def yt(r, r_, Q, s, action, n, UAVs, l, k, a, R, actionSpace, clusters, B, PU, g, l_, o2, useDL):
    (maxQ, _) = getMaxQ(s, action, n, UAVs, l, k, a, R, actionSpace, clusters, B, PU, g, l_, o2, useDL)
    return r + r_ * maxQ
    
# Q^pi(s, a) = E[Sum(k=0, inf)(r_^k * r_(t+k)) | st, at, pi]
#def Qpi(r_, r, t, k, s, a, pi):
#    # ( [6] get Q(s, a) using policy pi )
