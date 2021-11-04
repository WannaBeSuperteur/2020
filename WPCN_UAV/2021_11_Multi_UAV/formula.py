# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1

import math
import numpy as np

### A. SYSTEM MODEL
# L, l       : the number of clusters (l : each cluster)
# t, T       : specific flight period of UAVs, t in [0, T], for example, T = 1.0
# N          : flight period T ---> N+1 time slots
# kl, Kl     : device / device set in the l-th cluster
# wkl, ql(t) : wkl = (xkl, ykl, 0), ql(t) = (xl(t), yl(t), hl(t)), h_min <= hl(t) <= h_max
#              where w = [[l, k, xkl, ykl, 0], ...]  , shape = (L, ...)
#                and q = [[l, t, xlt, ylt, hlt], ...], shape = (L, N+1, ...)
# alpha      : proportion of downlink WPT in a period
# Vmax       : maximum speed of UAVs
# SN         : length of each subslot for uplink, SN = (1-a)*T/N

# (from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8950047&tag=1)
# To facilitate the analysis, the flight period T is discretized into N+1 time slots.
# Let us analyze the system within a specific flight period of the UAVs, represented as t in [0, T].

# find xkl, ykl using binary search
def find_wkl(w, k, l):
    maxDevices = 1000
    max_ = len(w)-1
    min_ = 0

    goal_lk = l * maxDevices + k

    # extreme case (no data)
    if len(w) == 0: return None

    # extreme case (l is too large to find the right cluster)
    if w[max_][0] * maxDevices + w[max_][1] <= goal_lk:
        return len(w)

    # binary search
    while True:
        mid_ = (max_ + min_) // 2
        mid_lk = w[mid_][0] * maxDevices + w[mid_][1]
        
        if mid_lk == goal_lk:
            return mid_
        elif max_ < min_: # do not exist
            return None
        elif mid_lk > goal_lk:
            max_ = mid_ - 1
        else:
            min_ = mid_ + 1

# formula (1)
def formula_01(q, l, n, Vmax, SN, N):
    q_ln        = q[l*(N+1) + n]    [2:]
    q_before_ln = q[l*(N+1) + (n-1)][2:]
    
    eucNorm = sum(q_in[i] * q_before_ln[i] for i in range(3))
    return eucNorm <= Vmax * SN

# theta_l,kl[n] : theta_l,kl[n] = sin^-1(hl[n] / d_l,kl[n])
#                 where d_l,kl[n] = sqrt((xl[n] - x_kl)^2 + (yl[n] - y_kl)^2 + hl[n]^2)
# b1, b2        : constant values representing environment influence
# s             : another constant value
# PLoS          : LoS probability
# PNLoS         : NLoS probability = 1.0 - PLoS

# formula (2, 3, 4) : for d_l,kl[n] = d_(l0),k(l1)[n]
def getDist(q, w, k, l0, l1, n, N):
    
    try:
        q_ln = q[l0*(N+1) + n][2:]       # [xl[n], yl[n], hl[n]]
        w_kl = w[find_wkl(w, k, l1)][2:] # [x_kl , y_kl , 0    ]
    except:
        print('w, k, l1:', k, l1)

    # d_l,kl[n]
    dist = math.sqrt(pow(q_ln[0] - w_kl[0], 2) + pow(q_ln[1] - w_kl[1], 2) + pow(q_ln[2], 2))
    return dist

# formula (2) : for PLoS = P_LoS(theta_(l0),k(l1)) and PNLoS
def formula_02(q, w, k, l0, l1, n, N, s, b1, b2, getPLoS):

    # get distance d_l,kl[n]
    dist = getDist(q, w, k, l0, l1, n, N)

    # theta_l,kl[n]
    q_ln = q[l0*(N+1) + n][2:] # [xl[n], yl[n], hl[n]]
    theta = math.asin(q_ln[2] / dist)

    if getPLoS:
        return b1 * pow(180 / math.pi * theta - s, b2)
    else:
        return 1.0 - b1 * pow(180 / math.pi * theta - s, b2)

# formula (3) : for L_l,kl
def formula_03(q, w, k, l, n, N, mu1, mu2, fc, c, alpha, isLoS):
    
    # get distance d_l,kl[n]
    dist = getDist(q, w, k, l, l, n, N)

    if isLoS:
        return mu1 * pow(4 * math.pi * fc * dist / c, alpha)
    else:
        return mu2 * pow(4 * math.pi * fc * dist / c, alpha)

# formula (4) : for g_l,kl[n] = g_(l0),k(l1)[n], using formula (2)
def formula_04(q, w, k, l0, l1, n, N, s, b1, b2, mu1, mu2, fc, c, alphaL):

    # (PLoS * mu1 + PNLoS * mu2)^-1
    PLoS  = formula_02(q, w, k, l0, l1, n, N, s, b1, b2, True)
    NPLoS = 1.0 - PLoS

    part1 = 1 / (PLoS * mu1 + NPLoS * mu2)

    # (K0 * dist)^-alpha
    K0    = 4 * math.pi * fc / c
    dist  = getDist(q, w, k, l0, l1, n, N)

    part2 = pow(K0 * dist, -alphaL)
    
    return part1 * part2

# al[0]     : denote downlink WET node of UAV l
#             equal 0 or 1, energy is transferred or not by the l-th UAV
# a_l,kl[n] : uplink WIT allocation between UAV l and IoT device kl at n-th time slot
#             equal 0 or 1, IoT device kl does communicate/or not with l-th UAV
# E_kl      : collected energy of each IoT device kl at period T
# E_kl[n]   : available energy of device kl in n-th time slot
# P^U_kl[j] : uplink power of device kl at n-th time slot

# in this code,
# al        : al[0] for each UAV l
# alkl      : a_l,kl[n] for each UAV l, device k and time slot n
#             where alkl = [[l0, k, l1, n, value], ...]

# find alkl value using binary search
def find_alkl(alkl, l0, k, l1, n):
    maxUAVs      = 100
    maxDevices   = 100
    maxTimeSlots = 3600
    
    max_ = len(alkl)-1
    min_ = 0

    # extreme case (no data)
    if len(alkl) == 0: return None

    goal_alkl = (l0 * (maxDevices * maxUAVs * maxTimeSlots) +
                 k  * (maxUAVs * maxTimeSlots) +
                 l1 * maxTimeSlots +
                 n)
    
    max_alkl = (alkl[max_][0] * (maxDevices * maxUAVs * maxTimeSlots) +
                alkl[max_][1] * (maxUAVs * maxTimeSlots) +
                alkl[max_][2] * maxTimeSlots +
                alkl[max_][3])

    # extreme case (l0 is too large to find the right cluster)
    if max_alkl <= goal_alkl:
        return len(alkl)

    # binary search
    while True:
        mid_ = (max_ + min_) // 2
        
        mid_alkl = (alkl[mid_][0] * (maxDevices * maxUAVs * maxTimeSlots) +
                    alkl[mid_][1] * (maxUAVs * maxTimeSlots) +
                    alkl[mid_][2] * maxTimeSlots +
                    alkl[mid_][3])

        if mid_alkl == goal_alkl:
            return mid_
        elif max_ < min_: # do not exist
            return None
        elif mid_alkl > goal_alkl:
            max_ = mid_ - 1
        else:
            min_ = mid_ + 1

# ng     : energy conversion efficiency of devices (= 0.1)
# PD     : transmission power for UAVs' downlink   (= 40 dBm)
# alphaL : proportion of downlink WPT in a period (temp value)
# alphaP : path loss exponent

# formula (6) : for E_kl, using formula (4)
def formula_06(q, w, l, k, ng, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, PD):
    Ekl = 0
    for i in range(L):
        gikl = formula_04(q, w, k, i, l, 0, N, s, b1, b2, mu1, mu2, fc, c, alphaL)
        Ekl += ng * alphaL * T * al[i] * gikl * PD
    return Ekl

# using Sum(j=1,N) a_l,kl[j] * SN * P^U_kl[j] <= E_kl     ... (8),
# suppose that     a_l,kl[j] * SN * P^U_kl[j] <= E_kl / N
#                                   P^U_kl[j] <= E_kl / (N * a_l,kl[j] * SN) for j in 1..N (only if a_l,kl[j] > 0)
# so, suppose that P^U_kl[j] = E_kl / (N * a_l,kl[j] * SN) if a_l,kl[j] == 1, otherwise 0
# for formula (7, 9)

# todo: "remove this function and use P^U_max = - 20 dBm instead"
def getPUkln(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):

    SN = (1 - alphaP)*T/N

    # for E_kl
    Ekl = formula_06(q, w, l, k, ng, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, PD)
    alkl_value = alkl[find_alkl(alkl, l, k, l, n)][4]

    if alkl_value > 0: return Ekl / (N * alkl_value * SN)
    return 0

# formula (7) : for E_kl[n], using formula (6)
def formula_07(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):

    SN = (1 - alphaP)*T/N
    Ekl = formula_06(q, w, l, k, ng, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, PD)
    result = Ekl

    for j in range(1, n):
        PU = getPUkln(q, w, l, k, j, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
        result -= alkl[j] * SN * PU

# get I_kl[n] = inference received by UAV l, for formula (9)
def getInferencekl(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):
    result = 0
    for j in range(L):
        if j == l: continue
        
        PU = getPUkln(q, w, j, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
        g = formula_04(q, w, k, l, j, n, N, s, b1, b2, mu1, mu2, fc, c, alphaL)
        result += PU * g
    return result

# formula (9) : for received SINR r_kl[n], using formula (4)
def formula_09(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):
    PU = getPUkln(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
    g = formula_04(q, w, k, l, l, n, N, s, b1, b2, mu1, mu2, fc, c, alphaL)
    
    inference = getInferencekl(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
    o2 = -110 # noise power spectral
    
    return PU * g / (inference + o2)

# formula (10) : instantaneous throughput R_kl[n], using formula (9)
def formula_10(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):
    SINR = formula_09(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
    B = 1000000 # bandwidth = 1 MHz
    return B * math.log(1.0 + SINR, 2)

# formula (11) : average throughput R_kl of IoT device kl of the flight cycle T, using formula (10)
def formula_11(q, w, l, k, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD):
    result = 0

    for n in range(N):
        alkl_value = alkl[find_alkl(alkl, l, k, l, n)][4]
        throughput = formula_10(q, w, l, k, n, ng, alphaP, alphaL, N, T, s, b1, b2, mu1, mu2, fc, c, L, al, alkl, PD)
        result += alkl_value * throughput

    return result / T
