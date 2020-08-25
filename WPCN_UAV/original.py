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
    eucN = pow(eucNorm([p[n][0]-x[k], p[n][1]-y[k]]), 2) # ||p[n]-uk||^2
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
        leftSide = eucNorm([p[n][0]-p[n-1][0], p[n][1]-p[n-1][1]]) # p[n]-p[n-1]
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
# 3-5. Rk,S >= Rmin for k in K ... (18)
#      Rk,S is average throughput of GT k
def checkCond18(n, N, t, k, ng, o2, PUL, pI, g0, x, y, HI, r, Rmin):
    for k in range(1, K+1):
        RkS = avgThroughputForGT(n, N, t, k, ng, o2, PUL, pI, g0, x, y, HI, r) # Rk,S
        if RkS < Rmin: return False
    return True

# 3-6. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= Sum(i=1,n-1){t[i][0]*g0*s*P^DL/(||pE[i]-uk||^2 + HE^2)^(r/2)} for n in ^N and k in K ... (19)
#   => Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
#      where GE[n][k] = g0/(||pE[i]-uk||^2 + HE^2)^(r/2)
def checkCond19(t, PUL, g0, s, PDL, pE, x, y, HE, r, N, K):
    NHat = Khat(N) # ^N

    for n in range(NHat): # for n in ^N
        for k in range(1, K+1): # for k in K

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1): leftSide += t[i][k] * PUL[i][k]

            # Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
            rightSide = 0
            for i in range(1, n):
                GE = getG(pE, g0, x, y, n, k, HE, r) # GE[n][k]
                rightSide += t[i][0] * GE * s * PDL

            if leftSide > rightSide: return False

    return True

# 3-7. ||px[n]-px[n-1]|| <= δN*vmax^x for x in {I,E} and n in ^N ... (20)
def checkCond20(pI, pE, n, T, N, vImax, vEmax):
    condForI = checkCond13(pI, n, T, N, vImax) # ||pI[n]-pI[n-1]|| <= δN*vmax^I for n in ^N
    condForE = checkCond13(pE, n, T, N, vEmax) # ||pE[n]-pE[n-1]|| <= δN*vmax^E for n in ^N

    if condForI == True and condForE == True: return True
    else: return False

# 3-8. px[0] == px[N] for x in {I,E} ... (21)
def checkCond21(pI, pE, N):
    condForI = checkCond14(pI, N) # pI[0] == pI[N]
    condForE = checkCond14(pE, N) # pE[0] == pE[N]

    if condForI == True and condForE == True: return True
    else: return False

### [ P1-NL ] - NONLINEAR ###
# 3-9. Rk >= Rmin for k in K ... (23)
def checkCond23(n, p, g0, x, y, H, r, ng, o2, T, t, Rmin):
    for k in range(1, K+1): # for k in K
        Rk = getRk(n, k, p, g0, x, y, H, r, ng, o2, T, t) # Rk
        if Rk < Rmin: return False
    return True

# 3-10. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= (1/δN)*Sum(i=1,n-1)E^NL[i][k] for n in ^N and k in K ... (24)
def checkCond24(t, PUL, T, N, ENL, K):
    sN = T/N # δN: length of each slot

    for n in range(NHat): # for n in ^N
        for k in range(1, K+1): # for k in K

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1): leftSide += t[i][k] * PUL[i][k]

            # Sum(i=1,n-1)(t[i][0]*GE[n][k]*s*P^DL)
            rightSide = 0
            for i in range(1, n): rightSide += ENL[i][k] # Sum(i=1,n-1)E^NL[i][k]
            rightSide /= sN # (1/δN)*Sum(i=1,n-1)E^NL[i][k]

            if leftSide > rightSide: return False

    return True

########################################
###                                  ###
###  4. formula about the solutions  ###
###                                  ###
########################################

### commonly used ####
# 4-pre0. e[n][k] = t[n][k]*P^UL[n][k]
def gete(t, PUL, n, k):
    return t[n][k]*PUL[n][k]

# 4-pre1. X[n][k] = (e[n][k])^2/z[n][k]
def getX(t, PUL, n, k, z):
    enk = gete(t, PUL, n, k)
    return pow(enk, 2)/z[n][k]

# 4-pre2. EL_k,LB[n](w[n],z[n][k] | ^w[n],^z[n][k])
def ELkLB(w, z, wHat, zHat, n, k):
    # FILL IN THE BLANK
    return None

# 4-pre3. A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k])
def Ank(e, z, eHat, zHat, n, k):
    # FILL IN THE BLANK
    return None

# 4-pre4. ||p[n]-uk||^2 <= array[n][k]^(2/r)
def pnUkArray(p, x, y, array, r):
    
    # ||p[n]-uk||^2
    eNorm = eucNorm([p[n][0]-x[k], p[n][1]-y[k]]) # ||pI[n]-uk||
    leftSide = pow(eNorm, 2)

    # (array[n][k])^(2/r)
    rightSide = pow(array[n][k], 2/r)

    if leftSide <= rightSide: return True
    return False

# 4-pre5. R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])
def getRkLB(n, k, z, PUL, zHat):
    # FILL IN THE BLANK
    return None

# 4-pre6. E^NL_k,LB[i](z[i][k] | ^z[i][k])
def getENLkLB(i, k, z, zHat):
    # FILL IN THE BLANK
    return None

# 4-0. (e[n][k])^2 <= P^ULmax*t[n][k] for n in N and k in K ... (28)
def checkCond28(PULmax, N, K, t, PUL):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K
            enk = gete(t, PUL, n, k) # e[n][k]
            if enk*enk > PULmax*t[n][k]: return False

    return True

# 4-1. ||p[n]-uk||^2+H^2 <= (z[n][k])^(2/r) for n in N and k in K ... (34)
def checkCond34(p, x, y, H, z, r):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K
            eucN = pow(eucNorm([p[n][0]-x[k], p[n][1]-y[k]]), 2) # ||p[n]-uk||^2
            if eucN*eucN + H*H <= pow(z[n][k], 2/r): return True
    return False

### [ P1.1A ] - INTEGRATED ###
# 4-2.       Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2/t[n][k])/(||p[n]-uk||^2 + H^2)^(r/2))
#         >= Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k])) ... (29)
# 4-2. =>    Sum(n=2,N)(t[n][k]/N)*log2(1 + (G[n][k]*ng/o^2)*(e[n][k])^2/t[n][k])
#         >= Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k]))
# G[n][k] = g0/(||p[n]-uk||^2 + H^2)^(r/2)
def checkCond29(N, t, k, ng, o2, PUL, p, g0, x, y, H, r, z):

    # Sum(n=2,N)(t[n][k]/N)*log2(1 + (G[n][k]*ng/o^2)*(e[n][k])^2/t[n][k])
    leftSide = 0
    for n in range(2, N+1):
        G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
        enk = gete(t, PUL, n, k) # e[n][k]
        leftSide += (t[n][k]/N)*math.log(1 + (G*ng/o2)*pow(enk, 2)/t[n][k], 2)

    # Sum(n=2,N)(t[n][k]/N)*log2(1 + ((g0*ng/o^2)*(e[n][k])^2)/(t[n][k]*z[n][k]))
    rightSide = 0
    for n in range(2, N+1):
        enk = gete(t, PUL, n, k) # e[n][k]
        rightSide += (t[n][k]/N)*math.log(1 + ((g0*ng/o2)*pow(enk, 2))/(t[n][k]*z[n][k]), 2)

    if leftSide >= rightSide: return True
    else: return False

# 4-3. Sum(i=1,n-1)t[i][0]*(g0*s*PDL)/(||p[n]-uk||^2 + H^2)^(r/2) >= Sum(i=1,n-1)g0*s*PDL*(w[i]^2/z[i][k]) ... (30)
#   => Sum(i=1,n-1)(t[i][0]*G[n][k]*s*PDL) >= Sum(i=1,n-1)g0*s*PDL*(t[i][0]/z[i][k])
def checkCond30(t, k, s, PDL, g0, z, p, x, y, H, r):

    # Sum(i=1,n-1)(t[i][0]*G[n][k]*s*PDL)
    leftSide = 0
    for i in range(1, n):
        G = getG(p, g0, x, y, n, k, H, r) # G[n][k]
        leftSide += t[i][0]*G*s*PDL

    # Sum(i=1,n-1)g0*s*PDL*(t[i][0]/z[i][k])
    rightSide = 0
    for i in range(1, n): rightSide += g0*s*PDL*t[i][0]/z[i][k]

    if leftSide >= rightSide: return True
    else: return False

# 4-4. (1/N)*Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*X[n][k]/t[n][k]) >= Rmin for k in K ... (31)
def checkCond31(N, g0, ng, o2, t, Rmin, K, PUL, z):
    for k in range(1, K+1): # for k in K
        
        # Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*X[n][k]/t[n][k])
        sumVal = 0
        for n in range(2, N+1):
            Xnk = getX(t, PUL, n, k, z) # X[n][k]
            sumVal += t[n][k]*math.log(1 + (g0*ng/o2)*Xnk/t[n][k], 2)
        sumVal /= N # (1/N)*sumVal

        if sumVal >= Rmin: return True

    return False

# 4-5. Sum(i=2,n)(e[i][k])^2 <= Sum(i=1,n-1)(EL_k,LB[i](w[i],z[i][k] | ^w[i],^z[i][k])) for n in ^N and k, in K... (37)
def checkCond37(N, K, t, PUL, w, z, wHat, zHat):
    NHat = Khat(N) # ^N
    
    for n in range(NHat): # for n in ^N
        for k in range(1, K+1): # for k in K

            # Sum(i=2,n)(e[i][k])^2
            leftSide = 0
            for i in range(2, n+1): leftSide += pow(gete(t, PUL, n, k), 2)

            # Sum(i=1,n-1)(EL_k,LB[i](w[i],z[i][k] | ^w[i],^z[i][k]))
            rightSide = 0
            for i in range(1, n): rightSide += ELkLB(w, z, wHat, zHat, i, k)

            if leftSide > rightSide: return False

    return True

# 4-6. X[n][k] <= A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k]) for n in N and k in K ... (38)
def checkCond38(N, K, t, PUL, e, z, eHat, zHat):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            # A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k])
            Xnk = getX(t, PUL, n, k, z) # X[n][k]
            A_nk = Ank(e, z, eHat, zHat, n, k) # A[n][k](e[n][k], z[n][k] | ^e[n][k], ^z[n][k])

            if Xnk > A_nk: return False

    return True

# 4-7. p*[n] = p^(q)[n] for n in N ... (39)
# 4-8. t*[n][0] = w^(q)[n]^2 and t*[n][k] = t^(q)[n][k] for n in N and k in K ... (40)
# 4-9. P^UL*[n][k] = (e^(q)[n][k])^2/t[n][k] for t[n][k] != 0, and 0 for t[n][k] == 0, for n in N and k in K ... (41)

### [ P2.1A ] - SEPARATED ###
# 4-10. (1/N)*Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*Y[n][k]/t[n][k])) >= Rmin for k in K ... (42)
def checkCond42(N, t, g0, ng, o2, Y, Rmin, K):
    for k in range(1, K+1): # for k in K

        # Sum(n=2,N)(t[n][k]*log2(1 + (g0*ng/o^2)*Y[n][k]/t[n][k]))
        sumVal = 0
        for n in range(2, N+1):
            sumVal += t[n][k]*math.log(1 + (g0*ng/o2)*Y[n][k]/t[n][k], 2)
        sumVal /= N # (1/N)*sumVal

        if sumVal < Rmin: return False

    return True

# 4-11. ||pI[n]-uk||^2 <= (zI[n][k])^(2/r) for n in N and k in K ... (45)
# 4-12. ||pE[n]-uk||^2 <= (zE[n][k])^(2/r) for n in N and k in K ... (46)
def checkCond45and46(pI, pE, x, y, zI, zE, r, N, K):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K
            pnUkCompareI = pnUkArray(pI, x, y, zI, r) # ||pI[n]-uk||^2 <= (zI[n][k])^(2/r)
            pnUkCompareE = pnUkArray(pE, x, y, zE, r) # ||pE[n]-uk||^2 <= (zE[n][k])^(2/r)

            if pnUkComparI == False or pnUkCompareE == False: return False

    return True

# 4-13. Sum(i=2,n)(e[i][k])^2 <= Sum(i=1,n-1)EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k]) for n in ^N and k in K ... (47)
def checkCond47(w, zE, wHat, zEHat, t, PUL, N, K):
    NHat = Khat(N) # ^N
    
    for n in range(NHat): # for n in ^N
        for k in range(1, K+1): # for k in K

            # Sum(i=2,n)(e[i][k])^2
            leftSide = 0
            for i in range(2, n+1): leftSide += pow(gete(t, PUL, i, k), 2) # (e[i][k])^2

            # Sum(i=1,n-1)EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k])
            rightSide = 0
            for i in range(1, n): rightSide += ELkLB(w, zE, wHat, zEHat, i, k) # EL_k,LB[i](w[i],zE[i][k] | ^w[i],^zE[i][k])

            if leftSide > rightSide: return False
            
    return True

# 4-14. Y[n][k] <= A[n][k](e[n][k],zI[n][k] | ^e[n][k],^zI[n][k]) for n in N and k in K ... (47)
def checkCond48(Y, e, zI, eHat, zIHat, N, K):
    for n in range(1, N+1): # for n in N
        for k in range(1, K+1): # for k in K

            leftSide = Y[n][k] # Y[n][k]
            rightSide = Ank(e, zI, eHat, zIHat, n, k) # A[n][k](e[n][k],zI[n][k] | ^e[n][k],^zI[n][k])

            if leftSide > rightSide: return False

    return True

### [ P1.2A ] - NONLINEAR ###
# 4-15. (1/N)*Sum(n=2,N)(t[n][k]*R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])) >= Rmin for k in K ... (56)
def checkCond56(N, t, z, PUL, zHat, Rmin, K):
    for k in range(1, K+1): # for k in K

        # Sum(n=2,N)(t[n][k]*R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k]))
        sumVal = 0
        for n in range(2, N+1):
            RkLB = getRkLB(n, k, z, PUL, zHat) # R_k,LB[n](z[n][k],P^UL[n][k] | ^z[n][k])
            sumVal += t[n][k] * RkLB
        sumVal /= N # (1/N)*sumVal

        if sumVal < Rmin: return False

    return True

# 4-16. Sum(i=2,n)(t[i][k]*P^UL[i][k]) <= (1/δN)*Sum(i=1,n-1)E^NL_k,LB[i](z[i][k] | ^z[i][k]) for n in ^N and k in K ... (57)
def checkCond57(t, PUL, T, N, z, zHat, K):
    nHat = Khat(N) # ^N

    for n in range(NHat): # for n in ^N
        for k in range(1, K+1): # for k in K

            # Sum(i=2,n)(t[i][k]*P^UL[i][k])
            leftSide = 0
            for i in range(2, n+1): leftSide += t[i][k]*PUL[i][k]

            # (1/δN)*Sum(i=1,n-1)E^NL_k,LB[i](z[i][k] | ^z[i][k])
            rightSide = 0
            for i in range(1, n): rightSide += getENLkLB(i, k, z, zHat) # E^NL_k,LB[i](z[i][k] | ^z[i][k])
            
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
    ###  7. execution  ###
    ###                ###
    ######################
    print('\n <<< 1. execution result >>>')
