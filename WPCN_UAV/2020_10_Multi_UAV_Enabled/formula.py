import math
import numpy as np

# definition of each notation:
# n      : index for time slot (n-th time slot)
# N      : number of time slots = N+1
# l      : (lowercase L) a UAV = a cluster
# L      : number of UAVs = number of clusters
# k      : k_l (k) is a device in K_l (cluster l)
# K      : IoT device K1, K2, ..., and KL
# T      : flight cycle
# t      : specific flight period of UAVs, t in [0, T]
# q      : q[l](t) = (x[l](t), y[l](t), h[l](t))
#          q[n][l] = (x[n][l], y[n][l], h[n][l])
# x      : x[l](t) = x[n][l]: the x value of UAV l at time t
# y      : y[l](t) = y[n][l]: the y value of UAV l at time t
# h      : h[l](t) = h[n][l]: the altutude(height) of UAV l at time t
# ng     : ng in (0, 1], RF-to-direct current (DC) energy conversion efficiency of each device
# alpha  : path loss exponent = 2
# alphaL : learning rate for DQN = 0.0001
# PLoS   : PLoS probability
# PNLoS  : PNLoS probability = 1 - PLoS
# theta  : theta[l][k_l] = sin^-1(h[n][l]/d[n][l][k_l])
#          -> elevation angle from IoT device k[l] to UAV l in n-th time slot
# b1     : environmental parameter = 0.36
# b2     : environmental parameter = 0.21
# S_     : ζ = another constant value determined by both antenna and environment
# u1     : additional path loss for LoS
# u2     : additional path loss for NLoS
# PD     : transmission power for UAV's downlink = 40dBm
# PU     : peak power for IoT devices' uplink transmit = -20dBm
# B      : Bandwidth = 1 MHz
# o2     : Noise power spectral = -110dBm

# compute q[n][l] = (x[n][l], y[n][l], h[n][l])
def getq(x, y, h):
    return [x, y, h]
def getq_nl(x, y, h, n, l):
    return [x[n][l], y[n][l], h[n][l]]

# get q[n][l] from UAVs (l-th UAV, at n-th time slot)

# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is THE POSITION for x, y and h at time t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...

# that is, in the form of UAV[uav_No][x/y/h][time]
def getqFromUAV(UAV, n):
    
    x = UAV[0][n]
    y = UAV[1][n]
    h = UAV[2][n]

    return [x, y, h]

# ||.|| (Euclidean norm)
def eucNorm(array):
    result = 0
    for i in range(len(array)): result += pow(array[i], 2)
    return math.sqrt(result)

# the distance between UAV l and device k_l
# d[n][l][k][l] = sqrt((x[n][l] - x[k_l])^2 + (y[n][l] - y[k_l])^2 + h[n][l]^2)
# d[n][l][k][j] = sqrt((x[n][l] - x[k_j])^2 + (y[n][l] - y[k_j])^2 + h[n][l]^2)
# k, l        : device k_l
# k, j        : device k_j
# x, y, and h : for each UAV (cluster) (moving, h>0), the value of x[n][l], y[n][l] and h[n][l]
# xd and yd   : for each device (not moving, h=0)
# clusters    : in the form of [list([[x of device 0, y of device 0], ...]), ...]
# n           : omitted because x, y and h denotes x[n][l], y[n][l] and h[n][l], respectively
def d_nlkl(l, k, clusters, x, y, h):

    thisDevice = clusters[l][k] # each device : [x of device 0, y of device 0]
    xd = thisDevice[0]
    yd = thisDevice[1]

    return math.sqrt(pow(x - xd, 2) + pow(y - yd, 2) + pow(h, 2))
    
# get theta value for getPLoS function
# theta[l][k_l][n] = sin^-1(h[n][l]/d[n][l][k_l])
# h : the value of h[n][l]
# n : omitted because h denotes h[n][l]
def getTheta(l, k, clusters, x, y, h):
    return math.asin(h / d_nlkl(l, k, clusters, x, y, h))

# probability of LoS : line-of-sight (PLoS) and probability of NLoS : non-line-of-sight (PNLoS)
# isNot              : get PLoS if False, get PNLoS if True
# clusters           : in the form of [list([[x of device 0, y of device 0], ...]), ...]

# PLoS(theta[l][k_l][n]) = b1*(180/pi * theta[l][k_l][n] - S_)^b2 ... (2)
# PNLoS = 1 - PLoS
def getPLoS(isNot, l, k, clusters, x, y, h, b1, b2, S_):
    
    thetaVal = getTheta(l, k, clusters, x, y, h) # theta[l][k_l][n]
    
    # thetaVal must not be negative because (180 / math.pi * thetaVal) should be positive
    if thetaVal < 0.0: thetaVal += math.pi * 2
    
    PLoS = b1 * pow(180.0 / math.pi * thetaVal - S_, b2)

    try:
        assert(not isinstance(PLoS, complex))
    except:
        print('PLoS is a complex !!')
        print(thetaVal, x, y, h)
        print(PLoS, b1, math.pi, S_, b2)
        print('to power:', 180 / math.pi * thetaVal - S_)
        print('powered:', b2)
        exit(0)

    if isNot == False: return PLoS # return PLoS
    else: return 1 - PLoS # return PNLoS

# the channel's power gain between UAV l and device k_l
# g[n][l][k_l] = [P_LoS*u1 + P_NLoS*u2]^-1 * (K0*d[n][l][k_l])^(-a) ... (4)
# where K0 = 4pi*fc/c
def g_nlkl(PLoS, u1, PNLoS, u2, fc, l, k, clusters, x, y, h, alpha):
    c = 300000000 # speed of light
    K0 = 4 * math.pi * fc / c
    result0 = 1 / (PLoS * u1 + PNLoS * u2)
    result1 = pow(K0 * d_nlkl(l, k, clusters, x, y, h), -alpha)

    try:
        assert(not isinstance(result0 * result1, complex))
    except:
        print('g_nlkl is a complex !!')
        print(PLoS, u1, PNLoS, u2)
        print(K0, d_nlkl(l, k, clusters, x, y, h), -alpha)
        exit(0)
    
    return result0 * result1

# the collected energy of each IoT device k_l at period T
# E[k_l] = Sum(i=1,L)(ng * alpha * T * a[0][i] * g[0][i][k_l] * P^D), ... (6)
#           where all l in deviceList, kl in K
def E_kl(deviceList, L, ng, alpha, T, a, g, k, PD):
    result = 0
    for i in range(1, L+1):
        result += ng * alpha * T * a[0][i] * g[0][i][k] * PD

    return result

# available energy E_[n][k_l] of device k_l in n-th time slot
# E_[n][k_l] = E[k_l] - Sum(j=1,n-1)(a[j][l][k_l] * SN * PU[j][k_l]) ... (7)
def E_nkl(n, l, k, SN, PU, L, ng, alpha, T, a, g, PD):
    result = E_kl(L, ng, alpha, T, a, g, k, PD)
    for j in range(1, n):
        result -= a[j][l][k] * SN * PU[j][l][k]

    return result

# instantaneous throughput R_[n][k_l] of IoT device k_l
# R_[n][k_l]  = B[k_l]*log2(1 + r[n][k_l]) ... (10)
#               where r[n][k_l] = P^U[n][k_l]*g[n][l][k_l] / (I_[n][k_l] + o^2)
# PU[n][l][k] : P^U[n][k_l]
# g           : g[n][l][k_l]
# B           : B[k_l]

# where I_[n][k_l] = Sum(j=1, j!=l, L)(P^U[n][k_j]*g[n][l][k_j])
def I_nkl(L, PU, n, l, k, g):
    result = 0
    for j in range(L):
        if j == l: continue # except for j!=l

        # do not compute for error cases where k >= (the number of devices in cluster j)
        try:
            result += PU[n][j][k] * g[n][l][k][j]
        except:
            pass

    return result

# compute throughput of R_[n][k_l]    
def R_nkl(L, B, n, l, k, PU, g, o2):

    # using inference received by UAV l from cluster j, j in L, j is not equal to l
    inference = I_nkl(L, PU, n, l, k, g)
    r = PU[n][l][k] * g[n][l][k][l] / (inference + o2)

    try:
        return B * math.log(1+r, 2)
    except:
        print('B:', B)
        print('r:', r)
        print(' - PU[n][l][k]:', PU[n][l][k])
        print(' - g[n][l][k][l]:', g[n][l][k][l])
        print(' - inference:', inference)
        print(' - o2:', o2)
        print('1+r:', 1+r)
        print('math.log(1+r, 2):', math.log(1+r, 2))
        print('final:', B * math.log(1+r, 2))

# the average throughput R[k_l] of IoT device k_l of the flight cycle T
# R[k_l] = (1/T) * Sum(n=1,T)(a[n][l][k_l] * R_[n][k_l]) ... (11)
# N == T for this case
def R_kl(L, T, l, k, a, B, n, PU, g, o2):
    result = 0
    for n in range(1, T+1):
        result += a[n][l][k] * R_nkl(L, B, n, l, k, PU, g, o2)
    return result / T

# location of UAV[l] at time t
# q[l](t) = (x[l](t), y[l](t), h[l](t))
# q[n][l] = [x[n][l], y[n][l], h[n][l]]
def q_nl(x, y, h, n, l):
    return [x[n][l], y[n][l], h[n][l]]

# length of each subslot for uplink (alpha is proportion of downlink WPT in a period)
# SN = (1-alpha)T/N 
def SN(alpha, T, N):
    return (1-alpha)*T/N
