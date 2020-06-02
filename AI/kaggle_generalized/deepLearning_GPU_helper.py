import math
import random
import deepLearning_GPU
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

import warnings
warnings.filterwarnings("error")

# Flatten
def flatten(array):
    result = []
    for i in range(len(array)):
        for j in range(len(array[0])): result.append(array[i][j])
    return result

# sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# inverse of sigmoid
# y = 1/(1+e^(-x))
# x = 1/(1+e^(-y))
# 1/x = 1+e^(-y)
# 1/x - 1 = e^(-y)
# ln(1/x - 1) = -y
# ln((1-x)/x) = -y
# ln(x/(1-x)) = y -> y = ln(x/(1-x))
def invSigmoid(x, highBound):
    try:
        preResult = x / (1 - x)
        result = math.log(preResult)
        return result
    except RuntimeWarning: # catch runtime warnings
        print('value of x is ' + str(x) + ' -> highBound = ' + str(highBound))
        if x < 0.5: return 0.0 # return value must be >= 0.0
        else: return highBound

# 2차원 배열의 일부를 복사한 후 Flatten시켜 반환하는 함수
# 원래 2차원 배열을 array라 할 때, array[i][j]에서 i, j를 각각 index i, index j라 하자.
# 이때 복사할 범위의 index i는 iStart ~ iEnd-1, index j는 jStart ~ jEnd-1이다.
# array      : 2차원 배열
# iStart     : array의 index i의 시작값
# iEnd       : array의 index i의 종료값
# jStart     : array의 index j의 시작값
# jEnd       : array의 index j의 종료값
# exceedValue: 가능한 범위를 넘어선 경우 적용할, 학습할 screen에서의 값
def arrayCopyFlatten(array, iStart, iEnd, jStart, jEnd, exceedValue):
    result = []
    for i in range(iStart, iEnd):
        for j in range(jStart, jEnd):
            if exceed(array, i, j): result.append(exceedValue) # 가능한 범위를 넘어서는 경우
            else: result.append(array[i][j])
    return result

# 2차원 배열의 전체를 복사하는 함수
def arrayCopy(array):
    height = len(array)
    width = len(array[0])
    result = [[0] * width for j in range(height)]
    
    for i in range(height):
        for j in range(width):
            result[i][j] = array[i][j]

    return result
