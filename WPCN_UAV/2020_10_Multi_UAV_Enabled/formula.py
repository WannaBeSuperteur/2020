import math
import numpy as np

# definition of each notation:
# n      : index for time slot (n-th time slot)
# N      : number of time slots = N+1
# l      : (lowercase L) a UAV = a cluster
# L      : number of UAVs = number of clusters
# k      : k[l] is a deivce in K[l]
# K      : IoT device K1, K2, ..., and KL
# t,T    : specific flight period of UAVs, t in [0, T]
# q      : q[l](t) = q[l][t] = (x[l](t), y[l](t), h[l](t))
#          q[n][l] = q[n][l] = (x[n][l], y[n][l], h[n][l])
# x      : x[l](t): the x value of UAV l at time t
# y      : y[l](t): the y value of UAV l at time t
# h      : h[l](t): the altutude(height) of UAV l at time t
# ng     : ng in (0, 1], RF-to-direct current (DC) energy conversion efficiency of each device
# alpha  : path loss exponent = 2
# alphaL : learning rate for DQN = 0.0001
# PLoS   : PLoS probability
# PNLoS  : PNLoS probability = 1 - PLoS
# theta  : theta[l][k[l]] = sin^-1(h[n][l]/d[n][l][k[l]])
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

# ||.|| (Euclidean norm)
def eucNorm(array):
    result = 0
    for i in range(len(array)): result += pow(array[i], 2)
    return math.sqrt(result)

# the distance between UAV l and device k[l]
# d[n][l][k[l]] = sqrt((x[n][l] - x[k[l]])^2 + (y[n][l] - y[k[l]])^2 + h[n][l]^2)
def d_nlkl(n, l, k, x, y, h):
    return math.sqrt(pow(x[n][l]-x[k[l]], 2) + pow(y[n][l]-y[k[l]], 2) + pow(h[n][l], 2))

# probability of LoS : line-of-sight (PLoS) and probability of NLoS : non-line-of-sight (PNLoS)
# isNot              : get PLoS if False, get PNLoS if True
# PLoS(theta[l][k[l]]) = b1*(180/pi * theta[l][k[l]] - S_)^b2 ... (2)
# NPLoS = 1 - PLoS
def getPLoS(isNot, theta, l, k, b1, b2, S_):
    thetaVal = theta[l][k[l]] # theta[l][k[l]]
    PLoS = b1 * pow(180 * math.pi * thetaVal - S_, b2)

    if isNot == False: return PLoS # return PLoS
    else: return 1 - PLoS # return PNLoS

# the channel's power gain between UAV l and device k[l]
# g[n][l][k[l]] = [P_LoS*u1 + P_NLoS*u2]^-1 * (K0*d[n][l][k[l]])^(-a) ... (4)
# where K0 = 4pi*fc/c
def g_nlkl(PLoS, u1, PNLoS, u2, fc, n, l, k, x, y, h, a):
    
    c = 300000000 # speed of light
    K0 = 4 * math.pi * fc / c
    result0 = 1 / (PLoS * u1 + PNLoS * u2)
    result1 = pow(K0 * d_nlkl(n, l, k, x, y, h), -a)

    return result0 * result1

# inference received by UAV l from cluster j, j in L(script letter), j != l
# I_[n][k[l]] = Sum(j=1, j!=l, L) P^U[n][k[j]] * g[n][j][k[j]] (gArray = g)
def getI(L, PU, n, k, l, gArray):
    result = 0
    for j in range(1, L+1):
        if j != l: result += PU[n][k[j]] * gArray[n][j][k[j]]
        
    return result

# the collected energy of each IoT device k[l] at period T
# E[k[l]] = Sum(i=1,L)(ng * alpha * T * a[0][i] * g[0][i][k[l]] * P^D), ... (6)
#           where all l in deviceList, kl in K
def E_kl(deviceList, L, ng, alpha, T, a, g, k, l, PD):
    result = 0
    for i in range(1, L+1):
        result += ng * alpha * T * a[0][i] * g[0][i][k[l]] * PD

    return result

# available energy E_[n][k[l]] of device k[l] in n-th time slot
# E_[n][k[l]] = E[k[l]] - Sum(j=1,n-1)(a[j][l][k[l]] * SN * PU[j][k[l]]) ... (7)
def E_nkl(n, k, l, SN, PU, L, ng, alpha, T, a, g, PD):
    result = E_kl(L, ng, alpha, T, a, g, k, l, PD)
    for j in range(1, n):
        result -= a[j][i][k[l]] * SN * PU[j][k[l]]

    return result

# instantaneous throughput R_[n][k[l]] of IoT device k[l]
# R_[n][k[l]] = B[k[l]]*log2(1 + r[n][k[l]]) ... (10)
#               where r[n][k[l]] = P^U[n][k[l]]*g[n][l][k[l]] / (I_[n][k[l]] + o^2)
def R_nkl(B, k, l, n, PU, g, I_, o2):
    r = PU[n][k[l]] * g[n][l][k[l]] / (I_[n][k[l]] + o2)
    return B[k[l]] * math.log(1+r, 2)

# the average throughput R[k[l]] of IoT device k[l] of the flight cycle T
# R[k[l]] = (1/T) * Sum(n=1,N)(a[n][l][k[l]] * R_[n][k[l]]) ... (11)
def R_kl(T, N, l, k, B, n, PU, g, I_, o2):
    result = 0
    for n in range(1, N+1):
        result += a[n][l][k[l]] * R_nkl(B, k, l, n, PU, g, I_, o2)
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
