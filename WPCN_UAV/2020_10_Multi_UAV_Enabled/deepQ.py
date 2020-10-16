import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main

# state: q[n][l], {a[n][l][k[l]]}, {R[n][k[l]]}
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
    maxQ = -999999 # max(a')Q(s', a')
    for a_ in range(len(actionSpace)):
        if useDL == True: QofNextStateAction = # ( [1] obtain Q(s', a') using deep learning )
        else: QofNextStateAction = 0

        # update max(a')Q(s', a')
        if QofNextStateAction > maxQ: maxQ = QofNextStateAction

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
