import sys
import math
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main
import formula as f
import random

# q : q[l](t) = (x[l](t), y[l](t), h[l](t))
#     q[n][l] = (x[n][l], y[n][l], h[n][l])

# state: q[n][l], {a[n][l][k_l]}, {R[n][k_l]}
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...
# that is, in the form of UAV[uav_No][x/y/h][time]
def getS(UAV, n, l, ac, R):

    # q[n][l]       (q[n][l])  : the location of UAV l (1d array - x, y and h)
    # ac[n][l][k_l] (ac[n][l]) : the number of times that each device communicates with UAV l (1d array, for all devices)
    # R[n][k_l]     (R[n])     : the average throughput of devices (for each device k),
    #                            in l-th cluster (1d array, for the devices in l-th cluster)
    q = f.getqFromUAV(UAV, n)
    
    return [q, ac[n][l], R[n]]

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
    return (action[0]-1)*9 + (action[1]-1)*3 + (action[2]-1)

# get max(a')Q(s', a') (s' = nextState, a' = a_)
def getMaxQ(s, a, n, l, R, actionSpace, clusters, B, PU, g, l_, o2):

    # get Q values for the action space of next state s'
    if useDL == True: rewardsOfActionsOfNextState = deepLearningQ_test(getNextState(s, a, n, l, R, clusters, B, PU, g, l_, o2))

    # find optimal action a' = a_ that is corresponding to max(a')Q(s', a')
    maxQ = -999999 # max(a')Q(s', a')
    for a_ in range(len(actionSpace)):

        # when using deep learning, Q value is the maximum value among rewards for actions on the next state
        # otherwise,                Q value is 0
        if useDL == True: QofNextStateAction = max(rewardsOfActionsOfNextState)
        else: QofNextStateAction = 0

        # update max(a')Q(s', a')
        if QofNextStateAction > maxQ: maxQ = QofNextStateAction

# convert state = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}] to "1d array with numeric values"
# q[n][l] : the location of UAV l = (x[n][l], y[n][l], h[n][l])
def stateTo1dArray(state):
    q = state[0]
    a = state[1]
    R = state[2]
    return q + a + R # because q[n][l], a[n][l], and R[n][l] are all 1d arraies

# deep Learning using Q table (training function)
# printed : print the detail?
def deepLearningQ_training(Q, deviceName, epoch, printed):
    
    # Q Table           = [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
    # convert to input  = converted version of [[s0], [s1], ...]
    #            output = original  version of [[q00, q01, ...], [q10, q11, ...], ...]

    # input array (need to convert original array [s0])
    inputData = []
    for i in range(len(Q)): inputData.append(stateTo1dArray(Q[i][0]))

    # output array (as original)
    outputData = []
    for i in range(len(Q)): outputData.append(Q[i][1])

    # save input and output array as file
    helper.saveArray('Q_input.txt', inputData)
    helper.saveArray('Q_output.txt', outputData)

    # train using deep learning and save the model (testInputFile and testOutputFile is None)
    # need: modelConfig.txt
    # DON'T NEED TO APPLY SIGMOID to training output data, because DL.deeplearning applies it
    DL.deepLearning('Q_input.txt', 'Q_output.txt', None, None, None,
                    None, 0.0, None, 'modelConfig.txt', deviceName, epoch, printed, 'deepQ_model')

# deep Learning using Q table (test function -> return reward values for each action)
def deepLearningQ_test(state):

    # convert state into 1d array
    stateArray = stateTo1dArray(state)

    # get reward values of the state
    # NEED TO APPLY INV-SIGMOID to test output data, because just getting model output
    trainedModel = deepLearning_GPU.deepLearningModel('deepQ_model', True)
    testO = deepLearning_GPU.modelOutput(trainedModel, stateArray)
    outputLayer = testO[len(testO)-1]

    # apply inverse sigmoid to output values
    for i in range(len(outputLayer)): # for each output data
        for j in range(len(outputLayer[0])): # for each value of output data
            outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])

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

# directReward : direct reward for the action
# alphaL       : learning rate
# r_           : discount factor
# useDL        : TRUE for getting reward using deep learning
#                FALSE for setting to 0
def updateQvalue(Q, s, a, directReward, alphaL, r_, n, l, R, useDL, clusters, B, PU, g, l_, o2):

    # obtain max(a')Q(s', a') (s' = nextState, a' = a_)
    actionSpace = getActionSpace()
    nextState = getNextState(s, a, n, l, R, clusters, B, PU, g, l_, o2)
    maxQ = getMaxQ(s, a, n, l, R, actionSpace, clusters, B, PU, g, l_, o2)
        
    # update Q value
    sFound = False # find corresponding state?
    for i in range(len(Q)):
        if Q[i][0][0] == s: # Q[i] = [[s0], [q00, q01, ...]], Q[i][0] = [s0], Q[i][0][0] = s0
            actionIndex = getActionIndex(a)
            Q[i][1][actionIndex] = (1 - alphaL)*Q[i][1][actionIndex] + alphaL*(directReward + r_*maxQ)

            sFound = True
            break

    # when corresponding state is not found
    if sFound == False:

        # create new action-reward array,
        # and modify the value with corresponding index to the q value for action a
        qs = []
        for i in range(27): qs.append(0)
        actionIndex = getActionIndex(a)
        qs[actionIndex] = (1 - alphaL)*qs[actionIndex] + alphaL*(directReward + r_*maxQ)

        # append the state-action-reward [[s], qs] to Q table
        Q.append([[s], qs])

# get angle for the location of UAV l q[n][l] and time slot (n-1) to (n)
# IN RADIAN
def getAngle(q, n):
    if n > 0:
        locBefore = q[n-1][l]
        locAfter = q[n][l]

        # x and y value at time t-1
        xBefore = q[n-1][0]
        yBefore = q[n-1][1]

        # x and y value at time t
        xAfter = q[n][0]
        yAfter = q[n][1]

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
# s : [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}]
# a : [x, y, z]
# n : time slot index
def getNextLocation(s, a, n):
    
    # get current location
    curX = s[n][0] # q[n][l][0]
    curY = s[n][1] # q[n][l][1]
    curH = s[n][2] # q[n][l][2]

    # ASSUMPTION:
    # x = -1 : UAV turns to the left (-45 degrees)
    # x = 1  : UAV turns to the right (45 degrees)
    # y = -1 : UAV files backward (-5 m)
    # y = 1  : UAV flies forward (5 m)
    # z = -1 : UAV descends (-1 m)
    # z = 1  : UAV rises (1 m)
    
    # a : action ([-1, -1, -1] to [1, 1, 1])
    x = a[0]
    y = a[1]
    z = a[2]

    # get angle (radian) between time n-1 and n
    lastAngle = 0 # set default angle as 0
    for n_ in range(n, 0, -1):
        q = f.getq(curX, curY, curH)
        lastAngle = getAngle(q, n_)
        if lastAngle != 'not moved': break # lastAngle is numeric value

    # when all 'not moved' for n, n-1, n-2, ..., 0, then set as default
    if lastAngle == 'not moved': lastAngle = 0

    # get next angle by x
    if x == -1: nextAngle = (lastAngle - math.pi/4) % (math.pi*2)
    elif x == 0: nextAngle = lastAngle
    elif x == 1: nextAngle = (lastAngle + math.pi/4) % (math.pi*2)

    # get next location by x and y
    # initialize as current location
    nextY = curX
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
# a       : action ([-1, -1, -1] to [1, 1, 1])

## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]
def getNextState(s, a, n, l, R, clusters, B, PU, g, l_, o2):

    # get next location
    [nextX, nextY, nextH] = getNextLocation(s, a, n)

    #### derive next a[n][1] ####
    # the number of times that each device communicates with UAV l

    #### derive next R[n][k_l] ####
    # the average throughput of devices in l-th cluster
    nextR = 0
    for k in range(len(clusters[l])): # for each device in cluster l
        nextR += f.R_nkl(B, k, l, n, PU, g, I_, o2)
    nextR /= len(clusters[l])

    #### return ####
    # s' = [q'[n][l], {a'[n][l][k_l]}, {R'[n][k_l]}]
    return [[nextX, nextY, nextH], next_a, nextR]

# target Q value yt = r + r_*max(a')Q(s', a', w) = r + r_*max(a')Q(s', a')
def yt(r, r_, Q, s, a, n, l, R, actionSpace, clusters, B, PU, g, l_, o2):
    maxQ = getMaxQ(s, a, n, l, R, actionSpace, clusters, B, PU, g, l_, o2)
    return r + r_ * maxQ
    
# Q^pi(s, a) = E[Sum(k=0, inf)(r_^k * r_(t+k)) | st, at, pi]
#def Qpi(r_, r, t, k, s, a, pi):
#    # ( [6] get Q(s, a) using policy pi )
