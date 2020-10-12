import math
import numpy as np

# ||.|| (Euclidean norm)
def eucNorm(array):
    result = 0
    for i in range(len(array)): result += pow(array[i], 2)
    return math.sqrt(result)

# d[n][l][k[l]] = sqrt((x[n][l] - x[k[l]])^2 + (y[n][l] - y[k[l]])^2 + h[n][l]^2)
def d_nlkl(n, l, k, x, y, h):
    return math.sqrt(pow(x[n][l]-x[k[l]], 2) + pow(y[n][l]-y[k[l]], 2) + pow(h[n][l], 2))

# E[k[l]] = Sum(i=1,L)(ng * alpha * T * a[0][i] * g[0][i][k[l]] * P^D), ... (6)
#           where all l in L, kl in K
def E_kl(L, ng, alpha, T, a, g, k, l, PD):
    result = 0
    for i in range(1, L+1):
        result += ng * alpha * T * a[0][i] * g[0][i][k[l]] * PD

    return result

# E_[n][k[l]] = E[k[l]] - Sum(j=1,n-1)(a[j][l][k[l]] * SN * PU[j][k[l]]) ... (7)
def E_nkl(n, k, l, SN, PU, L, ng, alpha, T, a, g, PD):
    result = E_kl(L, ng, alpha, T, a, g, k, l, PD)
    for j in range(1, n):
        result -= a[j][i][k[l]] * SN * PU[j][k[l]]

    return result

# R_[n][k[l]] = B[k[l]]*log2(1 + r[n][k[l]]) ... (10)
#               where r[n][k[l]] = P^U[n][k[l]]*g[n][l][k[l]] / (I[n][k[l]] + o^2)
def R_nkl(B, k, l, n, PU, g, I_, o2):
    r = PU[n][k[l]] * g[n][l][k[l]] / (I_[n][k[l]] + o2)
    return B[k[l]] * math.log(1+r, 2)

# R[k[l]] = (1/T) * Sum(n=1,N)(a[n][l][k[l]] * R_[n][k[l]]) ... (11)
def R_kl(T, N, l, k, B, n, PU, g, I_, o2):
    result = 0
    for n in range(1, N+1):
        result += a[n][l][k[l]] * R_nkl(B, k, l, n, PU, g, I_, o2)
    return result / T

# q[l](t) = (x[l](t), y[l](t), h[l](t))
# q[n][l] = [x[n][l], y[n][l], h[n][l]]
def q_nl(x, y, h, n, l):
    return [x[n][l], y[n][l], h[n][l]]

# SN = (1-a)T/N
def SN(alpha, T, N):
    return (1-alpha)*T/N
