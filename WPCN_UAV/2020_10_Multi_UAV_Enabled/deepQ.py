import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main

# state: q[n][l], {a[n][l][k_l]}, {R[n][k_l]}
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...
def getS(UAV, q, n, l, a, k, R):
    # ( [0] get state )

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
def getMaxQ(s, a, q, n, l, k, R, actionSpace):

    # get Q values for the action space of next state s'
    if useDL == True: rewardsOfActionsOfNextState = deepLearningQ_test(getNextState(s, a, q, n, l, k, R))

    # find optimal action a' = a_ that is corresponding to max(a')Q(s', a')
    maxQ = -999999 # max(a')Q(s', a')
    for a_ in range(len(actionSpace)):

        # when using deep learning, Q value is the maximum value among rewards for actions on the next state
        # otherwise,                Q value is 0
        if useDL == True: QofNextStateAction = max(rewardsOfActionsOfNextState)
        else: QofNextStateAction = 0

        # update max(a')Q(s', a')
        if QofNextStateAction > maxQ: maxQ = QofNextStateAction

# convert [state] = [q[n][l], {a[n][l][k_l]}, {R[n][k_l]}] to "1d array with numeric values"
def stateTo1dArray(state):
    # BLANK

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

# Q(s, a) <- (1 - alpha)*Q(s, a) + alpha*(Rwd(s, a) + lb*max(a')Q(s', a'))
# because using deep learning, just add Q value for this state and action

# when getting reward, for each UAV,
# 1. the UAV gets direct reward by the situation (in getReward function)
# 2. the state of UAV and the direct reward for the action is stored in the Q table
#    (for all other actions, save 0 as direct reward)
#    2-1. if there is a state that is the same with the state, update Q values for the state

# directReward : direct reward for the action
# useDL        : TRUE for getting reward using deep learning
#                FALSE for setting to 0
def updateQvalue(Q, s, a, directReward, alpha, lb, q, n, l, k, R, useDL):

    # obtain max(a')Q(s', a') (s' = nextState, a' = a_)
    actionSpace = getActionSpace()
    nextState = getNextState(s, a, q, n, l, k, R)
    maxQ = getMaxQ(s, a, q, n, l, k, R, actionSpace)
        
    # update Q value
    sFound = False # find corresponding state?
    for i in range(len(Q)):
        if Q[i][0][0] == s: # Q[i] = [[s0], [q00, q01, ...]], Q[i][0] = [s0], Q[i][0][0] = s0
            int actionIndex = getActionIndex(a)
            Q[i][1][a] = (1 - alpha)*Q[i][1][a] + alpha*(directReward + lb*maxQ)

            sFound = True
            break

    # when corresponding state is not found
    if sFound == False:

        # create new action-reward array,
        # and modify the value with corresponding index to the q value for action a
        qs = []
        for i in range(27): qs.append(0)
        int actionIndex = getActionIndex(a)
        qs[a] = (1 - alpha)*qs[a] + alpha*(directReward + lb*maxQ)

        # append the state-action-reward [[s], qs] to Q table
        Q.append([[s], qs])

# get next state
def getNextState(s, a, q, n, l, k, R):
    # ( [4] get next state )

# target Q value yt = r + r_*max(a')Q(s', a', w) = r + r_*max(a')Q(s', a')
def yt(r, r_, Q, s, a, q, n, l, k, R):
    maxQ = getMaxQ(s, a, q, n, l, k, R, actionSpace)
    return r + r_ * maxQ
    
# Q^pi(s, a) = E[Sum(k=0, inf)(r_^k * r_(t+k)) | st, at, pi]
def Qpi(r_, r, t, k, s, a, pi):
    # ( [6] get Q(s, a) using policy pi )
