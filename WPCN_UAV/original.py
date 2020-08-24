# UAV-Aided Wireless Powered Communication Networks: Trajectory Optimization and Resource Allocation for Minimum Throughput Maximization
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8836548

import math

##############################################################
###                                                        ###
###  0. formula (not from P1, P2, ... or their solutions)  ###
###                                                        ###
##############################################################

# 0-0. ||.|| (Euclidean norm)
def eucNorm(array):
    result = 0
    for i in range(len(array)): result += pow(array[i], 2)
    return math.sqrt(result)

# 0-1. ||p[n] - p[n-1]|| <= δN*v_max, n in {2,...,N} ... (2)
def maxSpeedConstraint(p, T, N, vmax):
    sN = T/N # δN: length of each slot

    # find Euclidean norm of p[n]-p[n-1] and check
    for n in range(2, N+1):
        array = [p[n][0]-p[n-1][0], p[n][1]-p[n-1][1]] # p[n]-p[n-1]
        eucN = eucNorm(array) # ||p[n]-p[n-1]||

        # return False if do not satisfy p[n]-p[n-1] <= δN*v_max
        if eucN > sN*vmax: return False

    return True

# 0-2. G[n][k] = g0/(||p[n]-uk||^2 + H^2)^(r/2) ... (3)
def getG(p, g0, x, y, n, k, H, r):
    eucN = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||p[n]-uk||^2
    return g0/pow(pow(eucN, 2) + H*H, r/2)

# 0-3. check condition for tk[n] ... (4)(5)
# refer form: t[n][k]
def checkForT(t):
    for n in range(1, N+1): # for n in N_

        # Sum(k=0,K) t[n][k] > 1 for n in N_
        if sum(tn) > 1: return False

        # 0 <= t[n][k] <= 1 for n in N_ and k in K_U{0}
        for k in range(K+1):
            if t[n][k] < 0 or t[n][k] > 1: return False

    return True

# 0-4. E^L[n][k] = t[n][0]*δN*Sk*G[n][k]*P^DL ... (6)
def getEL(n, k, T, N, s, G, PDL):
    sN = T/N # δN: length of each slot
    return t[n][0]*sN*s*G[n][k]*PDL

# 0-5. ~E[n][k] = Sum(i=1,n-1)E^L[i][k] - Sum(i=1,n-1)t[i][k]*δN*P^UL[i][k] ... (7)
def get_E(n, k, T, N, s, G, PDL, t, PUL):
    sN = T/N # δN: length of each slot
    
    # Sum(i=1,n-1)E^L[k][i]
    sum0 = 0
    for i in range(1, n): sum0 += getEL(i, k, T, N, s, G, PDL)

    # Sum(i=1,n-1)t[i][k]*δN*P^UL[i][k]
    sum1 = 0
    for i in range(1, n): sum1 += t[i][k]*sN*PUL[i][k]

    return sum0-sum1

# 0-6. check if t[n][k]*δN*P^UL[n][k] <= ~E[n][k] for n in ^N(Nhat) and k in K ... (8)
def uplinkPowerConstraint(n, k, T, N, s, G, PDL, t, PUL, K):
    sN = T/N # δN: length of each slot
    Nhat = Khat(N) # Nhat = {2,...,N}
    
    for n in Khat(N): # for n in ^N
        for k in K_(K): # for k in K_
            leftSide = t[n][k]*sN*PUL[n][k] # t[n][k]*δN*P^UL[n][k]
            rightSide = get_E(n, k, T, N, s, G, PDL, t, PUL) # ~E[n][k]
            if leftSide > rightSide: return False

    return True

# 0-7. R[n][k] = log2(1 + ngk*G[n][k]*PUL[n][k]/o^2) ... (9)
def getRnk(n, k, p, g0, x, y, H, r, ng, o2):
    G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
    return math.log(1 + ng*G*PUL[n][k]/o2, 2)

# 0-8. R[k] = (1/T) * δN*Sum(n=2,N)t[n][k]*R[n][k]
#           = (1/N) * Sum(n=2,N)t[n][k]*R[n][k]     ... (10)
def getRk(n, k, p, g0, x, y, H, r, ng, o2, T, t):

    # find Sum(n=2,N)t[n][k]*R[n][k]
    result = 0
    for n in range(2, N+1):
        Rnk = getRnk(n, k, p, g0, x, y, H, r, ng, o2) # R[n][k]
        result += t[n][k]*Rnk

    result = result / N # 1/N
    return result

# 0-9. uplink energy constraint of GT k at time slot n
#         t[n][k]*P^UL[n][k] <= Sum(i=1,n-1)[{(t[i][0]*s*g0*P^DL)/(||pE[i]-uk||^2 + HE^2)^(r/2)} - t[i][k]*P^UL[i][k]] ... (16)
#      => t[n][k]*P^UL[n][k] <= Sum(i=1,n-1)[(t[i][0]*s*GE[n][k]*P^DL) - t[i][k]*P^UL[i][k]]
#         where GE[n][k] = g0/(||pE[i]-uk||^2 + HE^2)^(r/2)
def uplinkEnergyConstraint(t, n, k, PUL, s, PDL, pE, g0, x, y, HE, r):

    # t[n][k]*P^UL[n][k]
    leftSide = t[n][k]*PUL[n][k]

    # Sum(i=1,n-1)[(t[i][0]*s*GE[n][k]*P^DL) - t[i][k]*P^UL[i][k]]
    rightSide = 0
    for i in range(1, n): # i = 1 to n-1
        GE = getG(pE, g0, x, y, n, k, HE, r) # GE[n][k]
        rightSide += t[i][0]*s*GE*PDL - t[i][k]*PUL[i][k]

    if leftSide <= rightSide: return True
    return False
    
# 0-10. average throughput of GT k Rk,S
#       = Sum(n=2,N)[(t[n][k]/N) * log2(1 + {(g0*ng/o^2)*P^UL[n][k]}/(||pI[n]-uk||^2 + HI^2)^(r/2)})] ... (17)
#       = Sum(n=2,N)[(t[n][k]/N) * log2(1 + (GI[n][k]*ng/o^2)*P^UL[n][k])]
#         where GI[n][k] = g0/(||pI[i]-uk||^2 + HI^2)^(r/2)
def avgThroughputForGT(n, N, t, k, ng, o2, PUL, pI, g0, x, y, HI, r):
    result = 0
    
    for n in range(2, N+1): # n = 2 to N
        GI = getG(pI, g0, x, y, n, k, HI, r) # GI[n][k]
        result += (t[n][k]/N) * math.log(1 + (GI*ng/o2)*PUL[n][k], 2)

    return result

# 0-11. harvested energy E^NL[n][k]
#       = (t[n][0]*δN/a)*{M(1+a)/(1+a*e^((-b*g0*P^DL)/(||p[n]-uk||^2 + H^2)^(r/2))) - M} ... (22)
#       = (t[n][0]*δN/a)*{M(1+a)/(1+a*e^(-b*G[n][k]*P^DL)) - M}
def harvestedEnergy(t, n, T, N, alpha, M, beta, PDL, p, g0, x, y, H, r):
    sN = T/N # δN: length of each slot
    G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
    expPart = math.exp((-1)*beta*G*PDL) # e^(-b*G[n][k]*P^DL)
    
    return (t[n][0]*sN/alpha) * (M*(1 + alpha)/(1 + alpha*expPart) - M)

###################################################
###                                             ###
###  1. other things (such as arrays and sets)  ###
###                                             ###
###################################################

# 1-0. K_ = {1,...,K} (also can used to get N_)
def K_(K):
    result = []
    for k in range(1, K+1): result.append(k)
    return result

# 1-1. Khat = {2,...,K} (also can used to get Nhat)
def Khat(K):
    result = []
    for k in range(2, K+1): result.append(k)
    return result

# 1-2. p[n] = p(nδN) = [xp(nδN), yp(nδN)]^T, where n in N_ ... (1)
# xFunc, yFunc : functions to get x_p(nδN) and y_p(nδN)
def pn(T, N, xFunc, yFunc):
    sN = T/N # δN: length of each slot

    p = [None] # p[0] = 0
    for n in range(1, N+1): # p[n] = [xp(nδN), yp(nδN)]^T
        p.append([xFunc(n*sN), yFunc(n*sN)])

    return p

# u[k] = [x[k], y[k]]^T (no need function for this)

#####################################
###                               ###
###  2. functions to get x and y  ###
###                               ###
#####################################

# to get x_p(t) and y_p(t)

# 2-0. function to get x
def xFunc():
    # FILL IN THE BLANK
    return -1

# 2-1. function to get y
def yFunc():
    # FILL IN THE BLANK
    return -1

#######################################
###                                 ###
###  3. formula about the problems  ###
###                                 ###
#######################################

### [ P1 ] - INTEGRATED ###
# 3-0. Rk >= Rmin for k in K ... (11)
def checkCond11(Rmin, K, n, p, g0, x, y, H, r, ng, o2, T, t):
    for k in range(1, K+1):
        Rk = getRk(n, k, p, g0, x, y, H, r, ng, o2, T, t) # get Rk
        if Rk < Rmin: return False
    return True

# 3-1. Sum(i=2,n)t[i][k]*P^UL[i][k] <= (1/δN)*Sum(i=1,n-1)E^L[i][k] for n in ^N and k in K ... (12)
def checkCond12(t, PUL, T, N, K, EL):
    sN = T/N # δN: length of each slot

    for n in range(2, N+1): # for n in ^N
        for k in range(1, K+1): # for k in K
        
            # Sum(i=2,n)t[i][k]*P^UL[i][k]
            leftSide = 0
            for i in range(2, n+1): leftSide += t[i][k] * PUL[i][k]

            # (1/δN)*Sum(i=1,n-1)E^L[i][k]
            rightSide = 0
            for i in range(1, n): rightSide += EL[i][k]/sN

            if leftSide > rightSide: return False

    return True

# 3-2. ||p[n]-p[n-1]|| <= δN*vmax for n in ^N ... (13)
def checkCond13(p, n, T, N, vmax):
    sN = T/N # δN: length of each slot

    for n in range(2, N+1): # for n in ^N
        leftSide = [p[n][0]-p[n-1][0], p[n][1]-p[n-1][1]] # p[n]-p[n-1]
        rightSide = sN*vmax # δN*vmax

        if leftSide > rightSide: return False

    return True

# 3-3. p[0] == p[N] ... (14)
def checkCond14(p, N):
    if p[0][0] == p[N][0] and p[0][1] == p[N][1]: return True
    return False

# 3-4. 0 <= P^UL[n][k] <= P^UL_max for n in N and k in K ... (15)
def checkCond15(PUL, n, k, PULmax, N, K):
    midSide = PUL[n][k] # P^UL[n][k]
    rightSide = PULmax # P^UL_max

    if 0 <= midSide and midSide <= rightSide: return True
    return False

### [ P2 ] - SEPARATED ###
# ... (18)

# ... (19)

# ... (20)

# ... (21)

### [ P1-NL ] - NONLINEAR ###
# ... (23)

# ... (24)

########################################
###                                  ###
###  4. formula about the solutions  ###
###                                  ###
########################################

### commonly used ####
# ... (28)

# ... (34)

### [ P1.1A ] - INTEGRATED ###
# ... (29)

# ... (30)

# ... (31)

# ... (37)

# ... (38)

# ... (39)

# ... (40)

# ... (41)

### [ P2.1A ] - SEPARATED ###
# ... (42)

# ... (45)

# ... (46)

# ... (47)

# ... (48)

### [ P1.2A ] - NONLINEAR ###
# ... (56)

# ... (57)

### [ P1.3 ] - TIME RESOURCE ALLOCATION ###
# (4)(5)(23)(24)

##########################
###                    ###
###  5. configuration  ###
###                    ###
##########################

if __name__ == '__main__':
    f = open('original_config.txt', 'r')
    config = f.readlines()
    f.close()

    # common constants
    K = int(config[1].split(' ')[0]) # K : number of GTs
    N = int(config[2].split(' ')[0]) # N : number of equal-length time slots
    T = float(config[3].split(' ')[0]) # T : pre-determined time period (sec)
    H = float(config[4].split(' ')[0]) # H : flight altitude of UAVs
    HI = float(config[5].split(' ')[0]) # HI : flight altitude of ID UAV
    HE = float(config[6].split(' ')[0]) # HE : flight altitude of ET UAV
    vmax = float(config[7].split(' ')[0]) # vmax : max speed for UAVs
    vImax = float(config[8].split(' ')[0]) # vImax : max speed for ID UAVs
    vEmax = float(config[9].split(' ')[0]) # vEmax : max speed for ET UAVs
    PDL = float(config[10].split(' ')[0]) # P^DL : downlink power
    PULmax = float(config[11].split(' ')[0]) # P^ULmax : maximum value for uplink power
    r = float(config[12].split(' ')[0]) # r : reference channel gain at distance of 1m and path-loss exponent, r>=2
    o2 = float(config[13].split(' ')[0]) # o^2 : noise variance
    ng = float(config[14].split(' ')[0]) # ng : portion of stored energy used for uplink info transmission, ng_k=ng for k in K, 0<ng<=1
    s = float(config[15].split(' ')[0]) # ζ : energy conversion efficiency (of GT k)
    Rmin = float(config[16].split(' ')[0]) # Rmin : to reflect minimum throughput among GTs (Rk>=Rmin)
    alpha = float(config[17].split(' ')[0]) # a : determined by circuit specifications (NON-LINEAR EH MODEL)
    beta = float(config[18].split(' ')[0]) # b : determined by circuit specifications (NON-LINEAR EH MODEL)
    M = float(config[19].split(' ')[0]) # M
    g0 = float(config[20].split(' ')[0]) # g0 : reference channel gain

    # options
    linear = config[23].split(' ')[0] # linear(l) non-linear(n)
    integ = config[24].split(' ')[0] # integrated(i) separated(s)

    # for LINEAR EH MODEL
    lenX = float(config[27].split(' ')[0]) # length for x (m)
    lenY = float(config[28].split(' ')[0]) # length for y (m)
    HstaticAP = float(config[29].split(' ')[0]) # H for static AP (m)
    iters = int(config[30].split(' ')[0]) # number of iteration

    # print value
    print(' <<< 0. config >>>')
    print(K, N, T, H, HI, HE, vmax, vImax, vEmax)
    print(PDL, PULmax, r, o2, ng, s, Rmin, alpha, beta, M, g0)
    print(linear, integ, lenX, lenY, HstaticAP, iters)

    ######################
    ###                ###
    ###  6. execution  ###
    ###                ###
    ######################
    print('\n <<< 1. execution result >>>')
