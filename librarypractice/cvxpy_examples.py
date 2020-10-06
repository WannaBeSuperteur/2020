import math
import numpy as np
import random
import cvxpy as cvx
from cvxpy import *

# ALL PROBLEMS AND CONSTRAINTS MUST FOLLOW DCP RULE
# valid problems    : Minimize(convex)
#                     Maximize(concave)
# valid constraints : affine == affine
#                     convex <= concave
#                     concave >= convex

def P1():
    # variables
    x = Variable(1)
    y = Variable(1)
    object_ = x + 3 * y
    objective = Maximize(object_)

    # constraints
    constraints = []
    constraints.append(x >= 0)
    constraints.append(x <= 5)
    constraints.append(y >= 0)
    constraints.append(y <= 5)

    # problem solving
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    print(' -------- Result --------')
    print('x=' + str(x.value) + ' y=' + str(y.value))
    print('result=' + str(result))

def P2():
    # predefined variables
    n = 30

    # variables
    a = Variable(n)
    object_ = sum(a)
    objective = Maximize(object_)

    # constraints
    constraints = []
    for i in range(n):
        constraints.append(a[i] >= 0)
        constraints.append(a[i] <= i*i)

    # problem solving
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    print(' -------- Result --------')
    print('a=' + str(a.value))
    print('result=' + str(result))

def P3():
    # predefined variables
    n = 30

    # variables
    a = Variable(n)
    b = Variable(n)
    object_ = sum(a+b)
    objective = Maximize(object_)

    # constraints
    constraints = []
    for i in range(n):
        constraints.append(a[i] >= 0)
        constraints.append(a[i] <= i + 10)
        constraints.append(b[i] >= 0)
        constraints.append(b[i] <= 2*i + 10)
        constraints.append(cvx.abs(a[i] - b[i]) <= 5)

    # problem solving
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    print(' -------- Result --------')
    print('a=' + str(a.value))
    print('b=' + str(b.value))
    print('result=' + str(result))

def P4():
    # predefined variables
    n = 6
    m = 6

    # variables
    a = Variable((m, n))
    object_ = cvx.sum(a)
    objective = Maximize(object_)

    # constraints
    constraints = []
    for i in range(m): # 0 <= a[i][j] <= ij, i=0,...,m-1, j=0,...,n-1
        for j in range(n):
            constraints.append(a[i][j] >= 0)
            constraints.append(a[i][j] <= i*j)
    for i in range(m): # 0 <= Sum(j=0, n-1)a[i][j] <= 2i, i=0,...,m-1
        constraints.append(sum(a[i]) >= 0)
        constraints.append(sum(a[i]) <= 2*i)
    for j in range(n): # 0 <= Sum(i=0, m-1)a[i][j] <= 2.5j, j=0,...,n-1
        constraints.append(sum(a.T[j]) >= 0)
        constraints.append(sum(a.T[j]) <= 2.5*j)

    # problem solving
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    print(' -------- Result --------')
    print('a=' + str(a.value))
    print('result=' + str(result))

def P5():
    # predefined variables
    n = 6

    # variables
    a = Variable((n, n))
    b = Variable(n)
    c = Variable(n)

    object_ = sum(b)
    objective = Maximize(object_)

    # constraints
    constraints = []
    for i in range(n): # 0 <= a[i][j] <= b[i] + c[j], i=0,...,n-1, j=0,...,n-1
        for j in range(n):
            constraints.append(a[i][j] >= 0)
            constraints.append(a[i][j] <= b[i] + c[j])
    for i in range(n): # 0 <= b[i] <= i, 0 <= c[i] <= 0.25 * b[i] + 0.25, |b[i]-c[i]| <= 5, i=0,...,n-1
        constraints.append(b[i] >= 0)
        constraints.append(b[i] <= i)
        constraints.append(c[i] >= 0)
        constraints.append(c[i] <= 0.25 * b[i] + 0.25)
        constraints.append(abs(b[i]-c[i]) <= 5)

    # problem solving
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    print(' -------- Result --------')
    print('a=' + str(a.value))
    print('b=' + str(b.value))
    print('c=' + str(c.value))
    print('result=' + str(result))

if __name__ == '__main__':
    P1()
    P2()
    P3()
    P4()
    P5()
