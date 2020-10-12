import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as main

# state: q[n][l], {a[n][l][k[l]]}, {R[n][k[l]]}
def getS(q, n, l, a, k, R):
    # ( [0] get state )

# get action space: from (-1, -1, -1) to (1, 1, 1)
def getActionSpace():
    actionSpace = []
    
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                actionSpace.append([x, y, z])

    return actionSpace

# update Q value

# Q Table: [[[s0], [q00, q01, ...]], [[s1], [q10, q11, ...]], ...]
# for new state-action-reward-nextState, get reward using deep learning,
# or set to 0 when amount of data is not big enoughly

# Q(s, a) <- (1 - alpha)*Q(s, a) + alpha*(Rwd(s, a) + lb*max(a')Q(s', a'))
# because using deep learning, just add Q value for this state and action

# useDL : TRUE for getting reward using deep learning
#         FALSE for setting to 0
def updateQvalue(Q, s, a, alpha, lb, q, n, l, k, R, useDL):

    # obtain max(a')Q(s', a') (s' = nextState, a' = a_)
    actionSpace = getActionSpace()
    nextState = getNextState(s, a, q, n, l, k, R)

    maxQ = -999999 # max(a')Q(s', a')
    for a_ in range(len(actionSpace)):
        if useDL == True: QofNextStateAction = # ( [1] obtain Q(s', a') using deep learning )
        else: QofNextStateAction = 0

        # update max(a')Q(s', a')
        if QofNextStateAction > maxQ: maxQ = QofNextStateAction

    # ( [2] update Q value )

# Rwd(s, a) - defined as 1) ~ 5) in the original paper 
def getReward(s, a):
    # ( [3] get reward value )

# get next state
def getNextState(s, a, q, n, l, k, R):
    # ( [4] get next state )

# yt = r + r_*max(a')Q(s', a', w)
def yt(r, r_, Q, s_, a_, w):
    # ( [5] get the value of yt )
    
# Q^pi(s, a) = E[Sum(k=0, inf)(r_^k * r_(t+k)) | st, at, pi]
def Qpi(r_, r, t, k, s, a, pi):
    # ( [6] get Q(s, a) using policy pi )
