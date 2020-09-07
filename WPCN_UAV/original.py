# UAV-Aided Wireless Powered Communication Networks: Trajectory Optimization and Resource Allocation for Minimum Throughput Maximization
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8836548

import math
import numpy as np
import random
from cvxpy import *

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
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def getG(p, g0, x, y, n, k, H, r):
    
    try: # p[dim0][dim1] form
        eucN = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||p[n]-uk||
    except: # p[dim0] form
        eucN = eucNorm([p[0]-x[k], p[1]-y[k]]) # ||p[n]-uk||
    
    return g0/pow(pow(eucN, 2) + H*H, r/2)

# 0-3. check condition for tk[n] ... (4)(5)
# refer form: t[n][k]
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkForT(t, n_, k_):
    for n in range(1, N+1): # for n in N_

        # Sum(k=0,K) t[n][k] > 1 for n in N_
        if sum(t[n]) > 1: return False

        # 0 <= t[n][k] <= 1 for n in N_ and k in K_U{0}
        for k in range(K+1):
            
            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue
            
            if t[n][k] < 0 or t[n][k] > 1: return False

    return True

# 0-4. E^L[n][k] = t[n][0]*δN*Sk*G[n][k]*P^DL ... (6)
def getEL(n, k, T, N, s, G, PDL):
    sN = T/N # δN: length of each slot
    
    try:
        return t[n][0]*sN*s*G[n][k]*PDL
    except:
        return t*sN*s*G*PDL

# 0-5. ~E[n][k] = Sum(i=1,n-1)E^L[i][k] - Sum(i=1,n-1)t[i][k]*δN*P^UL[i][k] ... (7)
def get_E(n, k, T, N, s, G, PDL, t, PUL):
    sN = T/N # δN: length of each slot
    
    # Sum(i=1,n-1)E^L[k][i]
    sum0 = 0
    for i in range(1, n): sum0 += getEL(i, k, T, N, s, G, PDL)

    # Sum(i=1,n-1)t[i][k]*δN*P^UL[i][k]
    sum1 = 0
    for i in range(1, n):
        try:
            sum1 += t[i][k]*sN*PUL[i][k]
        except:
            sum1 += t*sN*PUL

    return sum0-sum1

# 0-6. check if t[n][k]*δN*P^UL[n][k] <= ~E[n][k] for n in ^N(Nhat) and k in K ... (8)
def uplinkPowerConstraint(n, k, T, N, s, G, PDL, t, PUL, K):
    sN = T/N # δN: length of each slot
    Nhat = Khat(N) # Nhat = {2,...,N}
    
    for n in Khat(N): # for n in ^N
        for k in K_(K): # for k in K_
            try:
                leftSide = t[n][k]*sN*PUL[n][k] # t[n][k]*δN*P^UL[n][k]
            except:
                leftSide = t*sN*PUL # t[n][k]*δN*P^UL[n][k]

            rightSide = get_E(n, k, T, N, s, G, PDL, t, PUL) # ~E[n][k]
            if leftSide > rightSide: return False

    return True

# 0-7. R[n][k] = log2(1 + ngk*G[n][k]*PUL[n][k]/o^2) ... (9)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def getRnk(n, k, p, g0, x, y, H, r, ng, o2, PUL):
    G = getG(p, g0, x, y, n, k, H, r) # G[n][k]

    try:
        return math.log(1 + ng*G*PUL[n][k]/o2, 2)
    except:
        return math.log(1 + ng*G*PUL/o2, 2)

# 0-8. R[k] = (1/T) * δN*Sum(n=2,N)t[n][k]*R[n][k]
#           = (1/N) * Sum(n=2,N)t[n][k]*R[n][k]     ... (10)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def getRk(k, p, g0, x, y, H, r, ng, o2, t, PUL):

    # find Sum(n=2,N)t[n][k]*R[n][k]
    result = 0
    for n in range(2, N+1):
        Rnk = getRnk(n, k, p, g0, x, y, H, r, ng, o2, PUL) # R[n][k]

        try:
            result += t[n][k]*Rnk
        except:
            result += t*Rnk

    result = result / N # 1/N
    return result

# 0-9. uplink energy constraint of GT k at time slot n
#         t[n][k]*P^UL[n][k] <= Sum(i=1,n-1)[{(t[i][0]*s*g0*P^DL)/(||pE[i]-uk||^2 + HE^2)^(r/2)} - t[i][k]*P^UL[i][k]] ... (16)
#      => t[n][k]*P^UL[n][k] <= Sum(i=1,n-1)[(t[i][0]*s*GE[n][k]*P^DL) - t[i][k]*P^UL[i][k]]
#         where GE[n][k] = g0/(||pE[i]-uk||^2 + HE^2)^(r/2)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def uplinkEnergyConstraint(t, n, k, PUL, s, PDL, pE, g0, x, y, HE, r):

    # t[n][k]*P^UL[n][k]
    try:
        leftSide = t[n][k]*PUL[n][k]
    except:
        leftSide = t*PUL

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
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def avgThroughputForGT(N, t, k, ng, o2, PUL, pI, g0, x, y, HI, r):
    result = 0
    
    for n in range(2, N+1): # n = 2 to N
        GI = getG(pI, g0, x, y, n, k, HI, r) # GI[n][k]

        try:
            result += (t[n][k]/N) * math.log(1 + (GI*ng/o2)*PUL[n][k], 2)
        except:
            result += (t/N) * math.log(1 + (GI*ng/o2)*PUL, 2)

    return result

# 0-11. harvested energy E^NL[n][k]
#       = (t[n][0]*δN/a)*{M(1+a)/(1+a*e^((-b*g0*P^DL)/(||p[n]-uk||^2 + H^2)^(r/2))) - M} ... (22)
#       = (t[n][0]*δN/a)*{M(1+a)/(1+a*e^(-b*G[n][k]*P^DL)) - M}
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def harvestedEnergy(t, n, k, T, N, alpha, M, beta, PDL, p, g0, x, y, H, r):
    sN = T/N # δN: length of each slot
    G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
    expPart = math.exp((-1)*beta*G*PDL) # e^(-b*G[n][k]*P^DL)

    try:
        return (t[n][0]*sN/alpha) * (M*(1 + alpha)/(1 + alpha*expPart) - M)
    except:
        return (t*sN/alpha) * (M*(1 + alpha)/(1 + alpha*expPart) - M)

###################################################
###                                             ###
###  1. other things (such as arrays and sets)  ###
###                                             ###
###################################################

# distance between UAV p[n] whose location is p(nδN) = [xp(nδN), yp(nδN)]^T
# so actual location is (xp(t), yp(t), H),
# and GT k whose location is (x[k], y[k], 0)
# is eventually sqrt(||p[n] - uk||^2 + H^2) -> (||p[n] - uk||^2 + H^2)^r/2 = (dist)^r

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

# 2-0. function to get x according to time t
def xFunc(t):
    # CORE FUNCTION -> ORIGINAL OR USE DEEP LEARNING
    # FILL IN THE BLANK
    return -1 # temp

# 2-1. function to get y according to time t
def yFunc(t):
    # CORE FUNCTION -> ORIGINAL OR USE DEEP LEARNING
    # FILL IN THE BLANK
    return -1 # temp

#######################################
###                                 ###
###  3. formula about the problems  ###
###                                 ###
#######################################

### [ P1 ] - INTEGRATED ###
# 3-0. Rk >= Rmin for k in K ... (11)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# k_: check condition for only this k_ if specified (not None)
def checkCond11(Rmin, K, p, g0, x, y, H, r, ng, o2, t, PUL, k_):
    for k in range(1, K+1):
        
        # check for n_ and k_
        if k_ != None and k != k_: continue
        
        Rk = getRk(k, p, g0, x, y, H, r, ng, o2, t, PUL) # get Rk
        if Rk < Rmin: return False
    return True

# 3-1. Sum(i=2,n)t[i][k]*P^UL[i][k] <= (1/δN)*Sum(i=1,n-1)E^L[i][k] for n in ^N and k in K ... (12)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond12(t, PUL, T, N, K, EL, n_, k_):
    sN = T/N # δN: length of each slot

    for n in range(2, N+1): # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue
        
            # Sum(i=2,n)t[i][k]*P^UL[i][k]
            leftSide = 0
            for i in range(2, n+1):
                try: leftSide += t[i][k] * PUL[i][k]
                except: leftSide += t*PUL

            # (1/δN)*Sum(i=1,n-1)E^L[i][k]
            rightSide = 0
            for i in range(1, n):
                try: rightSide += EL[i][k]/sN
                except: rightSide += EL/sN

            if leftSide > rightSide: return False

    return True

# 3-2. ||p[n]-p[n-1]|| <= δN*vmax for n in ^N ... (13)
# pBefore: p[n-1] when p is p[n]
# n_     : check condition for only this n_ if specified (not None)
def checkCond13(p, pBefore, T, N, vmax, n_):
    sN = T/N # δN: length of each slot

    for n in range(2, N+1): # for n in ^N

        # check for n_ and k_
        if n_ != None and n != n_: continue

        try:
            leftSide = eucNorm([p[n][0]-p[n-1][0], p[n][1]-p[n-1][1]]) # p[n]-p[n-1]
        except:
            leftSide = eucNorm([p[0]-pBefore[0], p[1]-pBefore[1]]) # p[n]-p[n-1]
        rightSide = sN*vmax # δN*vmax

        if leftSide > rightSide: return False

    return True

# 3-3. p[0] == p[N] ... (14)
def checkCond14(p, p0, pN, N):
    if p0 != None and pN != None:
        if p[0][0] == p[N][0] and p[0][1] == p[N][1]: return True
    else:
        if p0[0] == pN[0] and p0[1] == pN[1]: return True
    return False

# 3-4. 0 <= P^UL[n][k] <= P^UL_max for n in N and k in K ... (15)
def checkCond15(PUL, n, k, PULmax):
    try:
        midSide = PUL[n][k] # P^UL[n][k]
    except:
        midSide = PUL
    
    rightSide = PULmax # P^UL_max

    if 0 <= midSide and midSide <= rightSide: return True
    return False

### [ P2 ] - SEPARATED ###
# 3-5. Rk,S >= Rmin for k in K ... (18)
#      Rk,S is average throughput of GT k
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# k_: check condition for only this k_ if specified (not None)
def checkCond18(N, t, ng, o2, PUL, pI, g0, x, y, HI, r, Rmin, k_):
    for k in range(1, K+1):

        # check for n_ and k_
        if k_ != None and k != k_: continue
            
        RkS = avgThroughputForGT(N, t, k, ng, o2, PUL, pI, g0, x, y, HI, r) # Rk,S
        if RkS < Rmin: return False
    return True

# 3-6. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= Sum(i=1,n-1){t[i][0]*g0*s*P^DL/(||pE[i]-uk||^2 + HE^2)^(r/2)} for n in ^N and k in K ... (19)
#   => Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
#      where GE[n][k] = g0/(||pE[i]-uk||^2 + HE^2)^(r/2)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond19(t, PUL, g0, s, PDL, pE, x, y, HE, r, N, K, n_, k_):
    NHat = Khat(N) # ^N

    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1):
                try: leftSide += t[i][k] * PUL[i][k]
                except: leftSide += t*PUL

            # Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
            rightSide = 0
            for i in range(1, n):
                GE = getG(pE, g0, x, y, n, k, HE, r) # GE[n][k]
                try: rightSide += t[i][0] * GE * s * PDL
                except: rightSide += t*GE*s*PDL

            if leftSide > rightSide: return False

    return True

# 3-7. ||px[n]-px[n-1]|| <= δN*vmax^x for x in {I,E} and n in ^N ... (20)
def checkCond20(pI, pE, n, T, N, vImax, vEmax):
    condForI = checkCond13(pI, None, T, N, vImax, None) # ||pI[n]-pI[n-1]|| <= δN*vmax^I for n in ^N
    condForE = checkCond13(pE, None, T, N, vEmax, None) # ||pE[n]-pE[n-1]|| <= δN*vmax^E for n in ^N

    if condForI == True and condForE == True: return True
    else: return False

# 3-8. px[0] == px[N] for x in {I,E} ... (21)
def checkCond21(pI, pI0, pIN, pE, pE0, pEN, N):
    condForI = checkCond14(pI, pI0, pIN, N) # pI[0] == pI[N]
    condForE = checkCond14(pE, pE0, pEN, N) # pE[0] == pE[N]

    if condForI == True and condForE == True: return True
    else: return False

### [ P1-NL ] - NONLINEAR ###
# 3-9. Rk >= Rmin for k in K ... (23)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# k_: check condition for only this k_ if specified (not None)
def checkCond23(p, g0, x, y, H, r, ng, o2, t, Rmin, PUL, k_):
    for k in range(1, K+1): # for k in K

        # check for n_ and k_
        if k_ != None and k != k_: continue
        
        Rk = getRk(k, p, g0, x, y, H, r, ng, o2, t, PUL) # Rk
        if Rk < Rmin: return False
    return True

# 3-10. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= (1/δN)*Sum(i=1,n-1)E^NL[i][k] for n in ^N and k in K ... (24)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond24(t, PUL, T, N, ENL, K, n_, k_):
    sN = T/N # δN: length of each slot
    NHat = Khat(N) # ^N

    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1):
                try: leftSide += t[i][k] * PUL[i][k]
                except: leftSide += t*PUL

            # Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
            rightSide = 0
            for i in range(1, n):
                try: rightSide += ENL[i][k] # Sum(i=1,n-1)E^NL[i][k]
                except: rightSide += ENL
            rightSide /= sN # (1/δN)*Sum(i=1,n-1)E^NL[i][k]

            if leftSide > rightSide: return False

    return True

########################################
###                                  ###
###  4. formula about the solutions  ###
###                                  ###
########################################

### commonly used ####
# 4-pre0. e[n][k] = sqrt(t[n][k]*P^UL[n][k]) ( >= 0 )
def gete(t, PUL, n, k):
    try:
        return math.sqrt(t[n][k]*PUL[n][k])
    except:
        return math.sqrt(t*PUL)

# 4-pre1. X[n][k] = (e[n][k])^2/z[n][k]
def getX(t, PUL, n, k, z):
    enk = gete(t, PUL, n, k)

    try:
        return pow(enk, 2)/z[n][k]
    except:
        return pow(enk, 2)/z

# 4-pre2. EL_k,LB[n](w[n],z[n][k] | ^w[n],^z[n][k]) ... (35)
#       = (g0*s*PDL*^w[n]/^z[n][k])(2w[n] - ^w[n] - ^w[n](z[n][k]-^z[n][k])/^z[n][k]) 
def getELkLB(w, z, wHat, zHat, n, k, g0, s, PDL):

    # g0*s*PDL*^w[n]/^z[n][k]
    try:
        part0 = g0*s*PDL*w[n]/z[n][k]
    except:
        part0 = g0*s*PDL*w/z

    # 2w[n] - ^w[n] - ^w[n](z[n][k]-^z[n][k])/^z[n][k]
    try:
        part1 = 2*w[n] - wHat[n] - wHat[n]*(z[n][k]-zHat[n][k])/zHat[n][k]
    except:
        part1 = 2*w - wHat - wHat*(z-zHat)/zHat
    
    return part0*part1

# 4-pre3. A[n][k](e[n][k],z[n][k] | ^e[n][k],^z[n][k]) ... (36)
#       = (^e[n][k])^2/^z[n][k] + (2*^e[n][k]/^z[n][k])(e[n][k]-^e[n][k]) - ((^e[n][k])^2/(^z[n][k])^2)(z[n][k] - ^z[n][k])
def getAnk(e, z, eHat, zHat, n, k):

    # (^e[n][k])^2/^z[n][k]
    try:
        part0 = pow(eHat[n][k], 2)/zHat[n][k]
    except:
        part0 = pow(eHat, 2)/zHat

    # (2*^e[n][k]/^z[n][k])(e[n][k]-^e[n][k])
    try:
        part1 = (2*eHat[n][k]/zHat[n][k])*(e[n][k] - eHat[n][k])
    except:
        part1 = (2*eHat/zHat)*(e - eHat)

    # ((^e[n][k])^2/(^z[n][k])^2)(z[n][k] - ^z[n][k])
    try:
        part2 = (pow(eHat[n][k], 2)/pow(zHat[n][k], 2))*(z[n][k] - zHat[n][k])
    except:
        part2 = (pow(eHat, 2)/pow(zHat, 2))*(z - zHat)

    return part0 + part1 - part2

# 4-pre4. ||p[n]-uk||^2 <= array[n][k]^(2/r)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def pnUkArray(p, n, k, x, y, array, r):
    
    # ||p[n]-uk||^2
    try:
        eNorm = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||pI[n]-uk||
    except:
        eNorm = eucNorm([p[0]-x[k], p[1]-y[k]])
    leftSide = pow(eNorm, 2)

    # (array[n][k])^(2/r)
    try:
        rightSide = pow(array[n][k], 2/r)
    except:
        rightSide = pow(array, 2/r)

    if leftSide <= rightSide: return True
    return False

# 4-pre5. R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])
# according to (54) in the paper, R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])
#       = log2(1 + (z[n][k] + g0*ng*P^UL[n][k]/o^2)/^z[n][k]) - (z[n][k] - ^z[n][k])/(^z[n][k]*ln2)
def getRkLB(n, k, z, PUL, zHat, g0, ng):

    # (z[n][k] + g0*ng*P^UL[n][k]/o^2)/^z[n][k]
    part0 = (z[n][k] + (g0*ng*PUL[n][k])/o2)/zHat[n][k]

    # (z[n][k] - ^z[n][k])/(^z[n][k]*ln2)
    part1 = (z[n][k] - zHat[n][k])/(zHat[n][k]*math.log(2))
    
    return math.log(1+part0, 2) - part1

# 4-pre6. E^NL_k,LB[n](z[n][k] | ^z[n][k]) ... (55)
#       = t[n][0]*M * {(1-exp(-^Z[n][k]))/(1+alpha*exp(-^Z[k][n]))
#         + ((1+alpha)*(^Z[n][k])^2*exp(-^Z[n][k])*(z[n][k]-^z[n][k]))/(beta*g0*P^DL*(1+alpha*exp(-^Z[n][k]))^2)}
#         + (Um/2)*(z[n][k]-^z[n][k])^2
#         where ^Z[n][k] = b*g0*P^DL/^z[n][k] and Um = min(z[n][k] in R)(U(z[n][k]))
# NOTE THAT ^Z is DIFFERENT from ^z
def getENLkLB(n, k, N, K, z, zHat, M, alpha, beta, g0, PDL):

    try:
        ZHat = beta*g0*PDL/z[n][k] #^Z[n][k] = b*g0*P^DL/^z[n][k]
        Um = U(z[0][0]) # Um = min(z[n][k] in R)(U(z[n][k]))
        for n in range(N+1):
            for k in range(K+1):
                if Um > U(z[n][k]): Um = U(z[n][k])
    except:
        ZHat = beta*g0*PDL/z
        Um = z

    # 1-exp(-^Z[n][k]))/(1+alpha*exp(-^Z[k][n])
    part0 = (1 - math.exp((-1)*ZHat))/(1 + alpha + math.exp((-1)*ZHat))

    # (1+alpha)*(^Z[n][k])^2*exp(-^Z[n][k])*(z[n][k]-^z[n][k])
    try:
        part1 = (1 + alpha)*pow(ZHat, 2)*math.exp((-1)*ZHat)*(z[n][k]-zHat[n][k])
    except:
        part1 = (1 + alpha)*pow(ZHat, 2)*math.exp((-1)*ZHat)*(z-zHat)

    # beta*g0*P^DL*(1+alpha*exp(-^Z[n][k]))^2
    part2 = beta*g0*PDL*pow(1 + alpha + math.exp((-1)*ZHat), 2)

    # (Um/2)*(z[n][k]-^z[n][k])^2
    try:
        part3 = (Um/2)*pow(z[n][k]-zHat[n][k], 2)
    except:
        part3 = (Um/2)*pow(z-zHat, 2)

    # E^NL_k,LB[n](z[n][k] | ^z[n][k])
    try:
        return t[n][0]*M*(part0 + part1/part2 + part3)
    except:
        return t*M*(part0 + part1/part2 + part3)

# 4-0. (e[n][k])^2 <= P^ULmax*t[n][k] for n in N and k in K ... (28)
def checkCond28(PULmax, N, K, t, PUL, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue
            
            enk = gete(t, PUL, n, k) # e[n][k]

            try:
                if enk*enk > PULmax*t[n][k]: return False
            except:
                if enk*enk > PULmax*t: return False

    return True

# 4-1. ||p[n]-uk||^2+H^2 <= (z[n][k])^(2/r) for n in N and k in K ... (34)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond34(p, x, y, H, z, r, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            try:
                eucN = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||p[n]-uk||
                if eucN*eucN + H*H <= pow(z[n][k], 2/r): return True
            except:
                eucN = eucNorm([p[0]-x[k], p[1]-y[k]]) # ||p[n]-uk||
                if eucN*eucN + H*H <= pow(z, 2/r): return True
    return False

### [ P1.1A ] - INTEGRATED ###
# 4-2.       Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2/t[n][k])/(||p[n]-uk||^2 + H^2)^(r/2))
#         >= Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k])) ... (29)
#      =>    Sum(n=2,N)(t[n][k]/N)*log2(1 + (G[n][k]*ng/o^2)*(e[n][k])^2/t[n][k])
#         >= Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k]))
# G[n][k] = g0/(||p[n]-uk||^2 + H^2)^(r/2)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# n_: check condition for only this n_ if specified (not None)
def checkCond29(N, t, k, ng, o2, PUL, p, g0, x, y, H, r, z, n_):

    # Sum(n=2,N)(t[n][k]/N)*log2(1 + (G[n][k]*ng/o^2)*(e[n][k])^2/t[n][k])
    leftSide = 0
    for n in range(2, N+1):

        # check for n_ and k_
        if n_ != None and n != n_: continue
        
        G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
        enk = gete(t, PUL, n, k) # e[n][k]

        try:
            leftSide += (t[n][k]/N)*math.log(1 + (G*ng/o2)*pow(enk, 2)/t[n][k], 2)
        except:
            leftSide += (t/N)*math.log(1 + (G*ng/o2)*pow(enk, 2)/t, 2)

    # Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k]))
    rightSide = 0
    for n in range(2, N+1):

        # check for n_ and k_
        if n_ != None and n != n_: continue
        
        enk = gete(t, PUL, n, k) # e[n][k]

        try:
            rightSide += (t[n][k]/N)*math.log(1 + ((g0*ng/o2)*pow(enk, 2))/(t[n][k]*z[n][k]), 2)
        except:
            rightSide += (t/N)*math.log(1 + ((g0*ng/o2)*pow(enk, 2))/(t*z), 2)

    if leftSide >= rightSide: return True
    else: return False

# 4-3. Sum(i=1,n-1)t[i][0]*(g0*s*PDL)/(||p[n]-uk||^2 + H^2)^(r/2) >= Sum(i=1,n-1)g0*s*PDL*(w[i]^2/z[i][k]) ... (30)
#   => Sum(i=1,n-1)(t[i][0]*G[n][k]*s*PDL) >= Sum(i=1,n-1)g0*s*PDL*(t[i][0]/z[i][k])
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def checkCond30(t, k, s, PDL, g0, z, p, x, y, H, r):

    # Sum(i=1,n-1)(t[i][0]*G[n][k]*s*PDL)
    leftSide = 0
    for i in range(1, n):
        G = getG(p, g0, x, y, n, k, H, r) # G[n][k]

        try: leftSide += t[i][0]*G*s*PDL
        except: leftSide += t*G*s*PDL

    # Sum(i=1,n-1)g0*s*PDL*(t[i][0]/z[i][k])
    rightSide = 0
    for i in range(1, n):
        try: rightSide += g0*s*PDL*t[i][0]/z[i][k]
        except: rightSide += g0*s*PDL*t/z

    if leftSide >= rightSide: return True
    else: return False

# 4-4. (1/N)*Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*X[n][k]/t[n][k]) >= Rmin for k in K ... (31)
# k_: check condition for only this k_ if specified (not None)
def checkCond31(N, g0, ng, o2, t, Rmin, K, PUL, z, k_):
    for k in range(1, K+1): # for k in K

        # check for n_ and k_
        if k_ != None and k != k_: continue
        
        # Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*X[n][k]/t[n][k])
        sumVal = 0
        for n in range(2, N+1):
            Xnk = getX(t, PUL, n, k, z) # X[n][k]

            try:
                sumVal += t[n][k]*math.log(1 + (g0*ng/o2)*Xnk/t[n][k], 2)
            except:
                sumVal += t*math.log(1 + (g0*ng/o2)*Xnk/t, 2)
        sumVal /= N # (1/N)*sumVal

        if sumVal >= Rmin: return True

    return False

# 4-5. Sum(i=2,n)(e[i][k])^2 <= Sum(i=1,n-1)(g0*s*P^DL*(w[i]^2/z[i][k])) for n in ^N and k in K ... (32)
def checkCond32(N, K, t, PUL, g0, s, PDL, w, z, n_, k_):
    NHat = Khat(N) # ^N
    
    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(e[i][k])^2
            leftSide = 0
            for i in range(2, n+1): leftSide += pow(gete(t, PUL, i, k), 2)

            # Sum(i=1,n-1)(g0*s*P^DL*(w[i]^2/z[i][k]))
            rightSide = 0
            for i in range(1, n):
                try: rightSide += g0*s*PDL*(w[i]*w[i]/z[i][k])
                except: rightSide += g0*s*PDL*(w*w/z)

            if leftSide > rightSide: return False

    return True

# 4-6. X[n][k] <= e[n][k]^2/z[n][k] for n in N and k in K ... (33)
def checkCond33(N, K, t, PUL, z, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue
            
            Xnk = getX(t, PUL, n, k, z) # X[n][k]
            enk = gete(t, PUL, n, k) # e[n][k]
            
            if Xnk > enk*enk/z[n][k]: return False

    return True

# 4-7. Sum(i=2,n)(e[i][k])^2 <= Sum(i=1,n-1)(EL_k,LB[i](w[i],z[i][k] | ^w[i],^z[i][k])) for n in ^N and k, in K ... (37)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond37(N, K, t, PUL, w, z, wHat, zHat, g0, s, PDL, n_, k_):
    NHat = Khat(N) # ^N
    
    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(e[i][k])^2
            leftSide = 0
            for i in range(2, n+1): leftSide += pow(gete(t, PUL, i, k), 2)

            # Sum(i=1,n-1)(EL_k,LB[i](w[i],z[i][k] | ^w[i],^z[i][k]))
            rightSide = 0
            for i in range(1, n): rightSide += getELkLB(w, z, wHat, zHat, i, k, g0, s, PDL)

            if leftSide > rightSide: return False

    return True

# 4-8. X[n][k] <= A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k]) for n in N and k in K ... (38)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond38(N, K, t, PUL, e, z, eHat, zHat, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k])
            Xnk = getX(t, PUL, n, k, z) # X[n][k]
            A_nk = getAnk(e, z, eHat, zHat, n, k) # A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k])

            if Xnk > A_nk: return False

    return True

# 4-9. p*[n] = p^(q)[n] for n in N ... (39)
# 4-10. t*[n][0] = w^(q)[n]^2 and t*[n][k] = t^(q)[n][k] for n in N and k in K ... (40)
# 4-11. P^UL*[n][k] = (e^(q)[n][k])^2/t[n][k] for t[n][k] != 0, and 0 for t[n][k] == 0, for n in N and k in K ... (41)

### [ P2.1A ] - SEPARATED ###
# 4-12. (1/N)*Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*Y[n][k]/t[n][k])) >= Rmin for k in K ... (42)
# k_: check condition for only this k_ if specified (not None)
def checkCond42(N, t, g0, ng, o2, Y, Rmin, K, k_):
    for k in range(1, K+1): # for k in K

        # check for n_ and k_
        if k_ != None and k != k_: continue

        # Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*Y[n][k]/t[n][k]))
        sumVal = 0
        for n in range(2, N+1):
            try:
                sumVal += t[n][k]*math.log(1 + (g0*ng/o2)*Y[n][k]/t[n][k], 2)
            except:
                sumVal += t*math.log(1 + (g0*ng/o2)*Y/t, 2)
        sumVal /= N # (1/N)*sumVal

        if sumVal < Rmin: return False

    return True

# 4-13. ||pI[n]-uk||^2 <= (zI[n][k])^(2/r) for n in N and k in K ... (45)
# 4-14. ||pE[n]-uk||^2 <= (zE[n][k])^(2/r) for n in N and k in K ... (46)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond45and46(pI, pE, x, y, zI, zE, r, N, K, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue
            
            pnUkCompareI = pnUkArray(pI, n, k, x, y, zI, r) # ||pI[n]-uk||^2 <= (zI[n][k])^(2/r)
            pnUkCompareE = pnUkArray(pE, n, k, x, y, zE, r) # ||pE[n]-uk||^2 <= (zE[n][k])^(2/r)

            if pnUkComparI == False or pnUkCompareE == False: return False

    return True

# 4-15. Sum(i=2,n)(e[i][k])^2 <= Sum(i=1,n-1)EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k]) for n in ^N and k in K ... (47)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond47(w, zE, wHat, zEHat, t, PUL, N, K, g0, s, PDL, n_, k_):
    NHat = Khat(N) # ^N
    
    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(e[i][k])^2
            leftSide = 0
            for i in range(2, n+1): leftSide += pow(gete(t, PUL, i, k), 2) # (e[i][k])^2

            # Sum(i=1,n-1)EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k])
            rightSide = 0
            for i in range(1, n): rightSide += getELkLB(w, zE, wHat, zEHat, i, k, g0, s, PDL) # EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k])

            if leftSide > rightSide: return False
            
    return True

# 4-16. Y[n][k] <= A[n][k](e[n][k],zI[n][k] | ^e[n][k],^zI[n][k]) for n in N and k in K ... (47)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond48(Y, e, zI, eHat, zIHat, N, K, n_, k_):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            try: leftSide = Y[n][k] # Y[n][k]
            except: leftSide = Y
            rightSide = getAnk(e, zI, eHat, zIHat, n, k) # A[n][k](e[n][k],zI[n][k] | ^e[n][k],^zI[n][k])

            if leftSide > rightSide: return False

    return True

### [ P1.2A ] - NONLINEAR ###
# 4-17. (1/N)*Sum(n=2,N)(t[n][k]*R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])) >= Rmin for k in K ... (56)
# k_: check condition for only this k_ if specified (not None)
def checkCond56(N, t, z, PUL, zHat, Rmin, K, g0, ng, k_):
    for k in range(1, K+1): # for k in K

        # check for n_ and k_
        if k_ != None and k != k_: continue

        # Sum(n=2,N)(t[n][k]*R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k]))
        sumVal = 0
        for n in range(2, N+1):
            RkLB = getRkLB(n, k, z, PUL, zHat, g0, ng) # R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])
            try: sumVal += t[n][k] * RkLB
            except: sumVal += t * RkLB
        sumVal /= N # (1/N)*sumVal

        if sumVal < Rmin: return False

    return True

# 4-18. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= (1/δN)*Sum(i=1,n-1)E^NL_k,LB[i](z[i][k] | ^z[i][k]) for n in ^N and k in K ... (57)
# n_, k_: check condition for only this n_ and k_ if specified (not None)
def checkCond57(t, PUL, T, N, z, zHat, K, M, alpha, beta, g0, PDL, n_, k_):
    nHat = Khat(N) # ^N

    for n in NHat: # for n in ^N
        for k in range(1, K+1): # for k in K

            # check for n_ and k_
            if n_ != None and k_ != None and (n != n_ or k != k_): continue

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1):
                try: leftSide += t[i][k]*PUL[i][k]
                except: leftSide += t*PUL

            # (1/δN)*Sum(i=1,n-1)E^NL_k,LB[i](z[i][k] | ^z[i][k])
            rightSide = 0
            for i in range(1, n): rightSide += getENLkLB(i, k, z, zHat, M, alpha, beta, g0, PDL) # E^NL_k,LB[i](z[i][k] | ^z[i][k])
            
            sN = T/N # δN: length of each slot
            rightSide /= sN # (1/δN)*...

            if leftSide > rightSide: return False

    return True

### [ P1.3 ] - TIME RESOURCE ALLOCATION ###
# (4)(5)(23)(24)

####################################
###                              ###
###  5. function for algorithms  ###
###                              ###
####################################
# Algorithm 1. Proposed Algorithm for (P1) With the Linear EH Model
def algorithm1(z, e, w, N, K, zHat, eHat, wHat, Rmin, p, t, PUL):
    q = 0
    zq = [] # refer form: zq[q][n][k] for q=0,1,2,..., n in N and k in K
    eq = [] # refer form: eq[q][n][k] for q=0,1,2,..., n in N and k in K
    tq = [] # refer form: tq[q][n][k] for q=0,1,2,..., n in N and k in K
    wq = [] # refer form: wq[q][n] for q=0,1,2,..., n in N
    pq = [] # refer form: pq[q][n] for q=0,1,2,..., n in N

    # set and solve (P1.1A) until convergence, for all n and all k
    while True:
        zq.append([[0 for k in range(K+1)] for n in range(N+1)])
        eq.append([[0 for k in range(K+1)] for n in range(N+1)])
        tq.append([[0 for k in range(K+1)] for n in range(N+1)])
        wq.append([0 for n in range(N+1)])
        pq.append([0 for n in range(N+1)])
        
        # setting values
        q += 1
        for n in range(0, N+1):
            for k in range(0, K+1):
                zHat[n][k] = zq[q-1][n][k]
                eHat[n][k] = eq[q-1][n][k]
                wHat[n] = wq[q-1][n]

        # X[n][k] = t[n][k]*PUL[n][k]/z[n][k] (proved that it is the optimum)
        X = [[0 for k in range(K+1)] for n in range(N+1)]
        for n in range(N+1):
            for k in range(K+1):
                X[n][k] = t[n][k]*PUL[n][k]/z[n][k]

        # solve P(1.1A) by using the CVX
        # reference: https://stackoverrun.com/ko/q/8432547
        objective = Maximize(Rmin)

        # solve the problem
        # constraints: (4)(5)(13)(14)(28)(31)(32)(33)(34)
        constraints = []

        p_ = Variable((N+1, 2), PSD=False) # p[n]
        Rmin_ = Variable() # R_min
        e_ = Variable((N+1, K+1), PSD=False) # e[n][k]
        t_ = Variable((N+1, K+1), PSD=False) # t[n][k]
        z_ = Variable((N+1, K+1), PSD=False) # z[n][k]
        w_ = Variable((N+1), PSD=False) # w[n]

        constraints.append(checkForT(t_, None, None)) # (4)(5) for n in N and k in {0,1,...,K}
        constraints.append(checkCond13(p_, None, T, N, vmax, None)) # (13) for n in ^N
        constraints.append(checkCond14(p_, None, None, N)) # (14)
        constraints.append(checkCond28(PULmax, N, K, t_, PUL, None, None)) # (28) for n in N and k in K
        for k in range(0, K+1):
            constraints.append(checkCond29(N, t_, k, ng, o2, PUL, p_, g0, x, y, H, r, z_, None)) # (29) for n in ^N
            constraints.append(checkCond30(t_, k, s, PDL, g0, z, p_, x, y, H, r)) # (30)
        constraints.append(checkCond31(N, g0, ng, o2, t_, Rmin_, K, PUL, z_, None)) # (31) for k in K
        constraints.append(checkCond34(p_, x, y, H, z_, r, None, None)) # (34) for n in N and k in K
        constraints.append(checkCond37(N, K, t_, PUL, w_, z_, wHat, zHat, g0, s, PDL, None, None)) # (37) for n in ^N and k in K

        prob = Problem(objective, constraints)
        result = prob.solve(verbose=True)

        # convergence check
        # FILL IN THE BLANK
        if convergence == True:

            # need (q+1) elements for arrays below
            zq.append([[0 for k in range(K+1)] for n in range(N+1)])
            eq.append([[0 for k in range(K+1)] for n in range(N+1)])
            tq.append([[0 for k in range(K+1)] for n in range(N+1)])
            wq.append([0 for n in range(N+1)])
            pq.append([0 for n in range(N+1)])
            
            break

    # set values
    pStar = [0 for n in range(N+1)] # p*[n]
    pULStar = [[0 for k in range(K+1)] for n in range(N+1)] # P^UL_k*[n]
    t0Star = [0 for n in range(N+1)] # t0*[n]
    tStar = [[0 for k in range(K+1)] for n in range(N+1)] # t*[n]
    
    for n in range(0, N+1):
        pStar[n] = pq[q][n] # p*[n] = p(q)[n]
        t0Star[n] = pow(w[q][n], 2) # t0*[n] = (w(q)[n])^2
        
        for k in range(0, K+1):
            PULStar[n][k] = pow(eq[q][n][k], 2)/t[n][k] # P^UL_k*[n] = (ek(q)[n])^2/t[n][k]
            tStar[n][k] = tq[n][k] # t*[n] = t(q)[n][k]

    return (pStar, t0Star, PULStar, tStar)

# Algorithm 2 Proposed Algorithm for (P1-NL) with the Non-Linear EH Model
def algorithm2(K, N, Rmin, PUL, p, z, t, T, M, alpha, beta, g0, ng, PDL):
    q = 0
    tq = [] # refer form: tq[q][n][k] for q=0,1,2,..., n in N and k in K
    pq = [] # refer form: pq[q][n] for q=0,1,2,..., n in N
    zqi = [] # refer form: zqi[q][i][n][k] for i, q, n in N and k in K
    pqi = [] # refer form: pqi[q][i][n] for i, q and n in N
    PULqi = [] # refer form: PULqi[q][i][n][k] for i, q, n in N and k in K
    zHat = [[0 for k in range(K+1)] for n in range(N+1)] # ^z[n][k]
    Rminq = Rmin # R^(q)_min

    # to prevent index error
    tq.append([[0 for k in range(K+1)] for n in range(N+1)])
    pq.append([[0, 0] for n in range(N+1)])
    zqi.append([]) # append zqi[q]
    pqi.append([]) # append pqi[q]
    PULqi.append([]) # append PULqi[q]

    # until Rminq converges
    while True:
        tq.append([[0 for k in range(K+1)] for n in range(N+1)])
        pq.append([[0, 0] for n in range(N+1)])
        zqi.append([]) # append zqi[q]
        pqi.append([]) # append pqi[q]
        PULqi.append([]) # append PULqi[q]

        # update
        q = q + 1
        i = 0

        # set z^(q,i)_k[n] = ||p^(q-1)[n] - uk||^2 for all n in N and all k in K
        zqi[q].append([[0 for k in range(K+1)] for n in range(N+1)]) # append zqi[q][i][n][k] (zqi[q][i] = ([n][k]))
        pqi[q].append([0 for n in range(N+1)]) # append pqi[q][i][n] ((pqi[q][i] = ([n]))
        PULqi[q].append([[0 for k in range(K+1)] for n in range(N+1)]) # append PULqi[q][i][n][k] (PULqi[q][i] = ([n][k]))
        
        for n in range(N+1):
            for k in range(K+1):
                zqi[q][i][n][k] = pow(eucNorm([pq[q-1][n][0]-x[k], pq[q-1][n][1]-y[k]]), 2)

        # repeat until convergence
        while True:
            
            # set ^z[n][k] = z^(q,i)[n][k]
            for n in range(N+1):
                for k in range(K+1):
                    zHat[n][k] = zqi[q][i][n][k]
            
            # solve (P1.2A) for given {t(q-1)[n][k]}
            # reference: https://stackoverrun.com/ko/q/8432547
            objective = Maximize(Rmin)

            # solve the problem
            # constraints: (13)(14)(15)(34)(56)(57)
            constraints = []

            p_ = Variable((N+1, 2), PSD=False) # p[n]
            Rmin_ = Variable() # R_min
            PUL_ = Variable((N+1, K+1), PSD=False) # P^UL[n][k]
            z_ = Variable((N+1, K+1), PSD=False) # z[n][k]

            constraints.append(checkCond13(p_, None, T, N, vmax, None)) # (13) for n in ^N
            constraints.append(checkCond14(p_, None, None, N)) # (14)
            for n in range(N+1):
                for k in range(K+1):
                    constraints.append(checkCond15(PUL_, n, k, PULmax)) # (15) for n in N and k in K
            constraints.append(checkCond34(p_, x, y, H, z_, r, None, None)) # (34) for n in N and k in K
            constraints.append(checkCond56(N, t, z_, PUL_, zHat, Rmin_, K, g0, ng, None)) # (56) for k in K
            constraints.append(checkCond57(t, PUL_, T, N, z_, zHat, K, M, alpha, beta, g0, PDL, None, None)) # (57) for n in ^N and k in K

            prob = Problem(objective, constraints)
            result = prob.solve(verbose=True)

            # update: i <- i+1
            i += 1
            
            zqi[q].append([[0 for k in range(K+1)] for n in range(N+1)]) # append zqi[q][i][n][k] (zqi[q][i] = ([n][k]))
            pqi[q].append([0 for n in range(N+1)]) # append pqi[q][i][n] ((pqi[q][i] = ([n]))
            PULqi[q].append([[0 for k in range(K+1)] for n in range(N+1)]) # append PULqi[q][i][n][k] (PULqi[q][i] = ([n][k]))

            # convergence check
            # FILL IN THE BLANK
            if convergence == True: break

        # p^(q)[n] = p^(q,i)[n] and P^UL(q)[n][k] = P^UL(q,i)[n][k]
        for n in range(N+1):
            for k in range(K+1):
                pq[q][n] = pqi[q][i][n]
                PUL[q][n][k] = PULqi[q][i][n][k]

        # compute R(q)_min and {t(q)[n][k]} from (P1.3)
        # FILL IN THE BLANK

        # convergence heck for R^(q)_min
        # FILL IN THE BLANK
        if convergence == True: return

# time allocation algorithm
# (P1.3) max(Rmin,{t[n][k]}) s.t. (4)(5)(23)(24)
# location of GT: in the form of [[x[1], y[1]], [x[2], y[2]], ..., [x[k], y[k]]]
def algorithmTimeAllocate(Rmin, t, n, k, p, g0, x, y, H, r, ng, o2, T, PUL, N, K, alpha, M, beta, PDL):

    # create E^NL array
    # harvested energy E^NL[n][k] = (t[n][0]*δN/a)*{M(1+a)/(1+a*e^((-b*g0*P^DL)/(||p[n]-uk||^2 + H^2)^(r/2))) - M} ... (22)
    ENL = [[0 for k_ in range(K+1)] for n_ in range(N+1)] # E^NL
    for n_ in range(1, N+1):
        for k_ in range(1, K+1):
            ENL[n_][k_] = harvestedEnergy(t, n_, k_, T, N, alpha, M, beta, PDL, p, g0, x, y, H, r)

    # solve (P1.3A)
    # reference: https://stackoverrun.com/ko/q/8432547
    objective = Maximize(Rmin)

    # solve the problem
    constraints = []

    Rmin_ = Variable() # R_min
    t_ = Variable() # t[n][k]
            
    constraints.append(checkForT(t_, None, None)) # (4)(5) for n in N and k in {0,1,...,K}
    constraints.append(checkCond23(p, g0, x, y, H, r, ng, o2, t_, Rmin_, PUL, None)) # (23) for k in K
    constraints.append(checkCond24(t_, PUL, T, N, ENL, K, None, None)) # (24) for n in ^N and k in K

    prob = Problem(objective, constraints)
    result = prob.solve(verbose=True)

##########################
###                    ###
###  6. configuration  ###
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

    # P^UL (MUST BE <=PULmax)
    PUL = [[0 for k in range(K+1)] for n in range(N+1)]
    f = open('PUL.txt', 'r')
    puls = f.readlines() # with n row(line)s and k columns
    f.close()
    
    for n in range(N): # for each row (n rows)
        thisLine = puls[n] # P^UL[n]
        thisLineSplit = thisLine.split('\n')[0].split(' ') # each element of P^UL[n]
        
        for k in range(K): # for each cell(k columns) in each row
            PUL[n+1][k+1] = float(thisLineSplit[k])
            assert(PUL[n+1][k+1] <= PULmax)

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
    print('')
    print(' <<< P^UL >>>')
    for i in range(1, N+1): print(PUL[i][1:K+1])
    print('')

    ######################
    ###                ###
    ###  7. execution  ###
    ###                ###
    ######################

    ### 7-0. location of GT k (k=1,2,...,K) -> place RANDOMLY
    x = [] # x axis for each GT k
    y = [] # y axis for each GT k
    x.append(-1) # x[0] = -1
    y.append(-1) # y[0] = -1
    for k in range(K): # randomly place GT within lenX and lenY
        x.append(random.random() * lenX)
        y.append(random.random() * lenY)

    ### 7-1. get p[n] first (using xFunc and yFunc function)
    p = pn(T, N, xFunc, yFunc)

    ### 7-2. let z[n][k] = (||p[n]-uk||^2 + H^2)^(r/2)
    z = [[0 for k in range(K+1)] for n in range(N+1)]
    for k in range(1, K+1):
        for n in range(1, N+1):
            eucN = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||p[n]-uk||
            z[n][k] = pow(eucN*eucN + H*H, r/2)

    ### 7-3. t = t[n][k] [NOT COMPLETED]
    t = [[0 for k in range(K+1)] for n in range(N+1)]
    for k in range(1, K+1):
        for n in range(1, N+1):
            t[n][k] = 0 # TEMP (find the value in the future)

    ### 7-4. e = e[n][k] = sqrt(t[n][k]*P^UL[n][k]) ( >= 0 )
    e = [[0 for k in range(K+1)] for n in range(N+1)]
    for k in range(1, K+1):
        for n in range(1, N+1):
            e[n][k] = gete(t, PUL, n, k)

    ### 7-5. w = w[n] = sqrt(t[n][0]) because t[n][0] = (w[n])^2
    w = [0 for n in range(N+1)]
    for n in range(1, N+1): w[n] = math.sqrt(t[n][0])

    ### 7-6. initialize Hat values (zHat, eHat and wHat) as original values
    zHat = [[0 for k in range(K+1)] for n in range(N+1)]
    eHat = [[0 for k in range(K+1)] for n in range(N+1)]
    wHat = [0 for n in range(N+1)]
            
    for n in range(1, N+1):
        wHat[n] = w[n]
        for k in range(1, K+1):
            zHat[n][k] = z[n][k]
            eHat[n][k] = e[n][k]
            
    ### 7-7. select algorithm
    algo = int(input('algorithm'))

    # Proposed Algorithm for (P1) With the Linear EH Model
    if algo == 1:
        print('\n <<< 1. execution result of algorithm 1 >>>')
        algorithm1(z, e, w, N, K, zHat, eHat, wHat, Rmin, p, t, PUL)

    # Proposed Algorithm for (P1-NL) with the Non-Linear EH Model
    elif algo == 2:
        print('\n <<< 2. execution result of algorithm 2 >>>')
        algorithm2(K, N, Rmin, PUL, p, z, t, T, M, alpha, beta, g0, ng, PDL)

    # time allocation algorithm
    elif algo == 3:
        print('\n <<< 3. execution result of time allocation algorithm >>>')
        algorithmTimeAllocate(Rmin, t, n, k, p, g0, x, y, H, r, ng, o2, T, PUL, N, K, alpha, M, beta, PDL)
