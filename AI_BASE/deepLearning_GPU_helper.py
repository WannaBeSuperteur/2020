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

# argmax
def argmax(array):
    maxValue = array[0]
    maxIndex = 0
    for i in range(len(array)):
        if array[i] > maxValue:
            maxValue = array[i]
            maxIndex = i
    return maxIndex

# inverse of sigmoid
# y = 1/(1+e^(-x))
# x = 1/(1+e^(-y))
# 1/x = 1+e^(-y)
# 1/x - 1 = e^(-y)
# ln(1/x - 1) = -y
# ln((1-x)/x) = -y
# ln(x/(1-x)) = y -> y = ln(x/(1-x))
def invSigmoid(x):
    try:
        preResult = x / (1 - x)
        result = math.log(preResult)
        return result
    except RuntimeWarning: # catch runtime warnings
        print('value of x is ' + str(x))
        if x < 0.5: return -15
        else: return 15

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

# copy 2d array
def arrayCopy(array):
    height = len(array)
    width = len(array[0])
    result = [[0] * width for j in range(height)]
    
    for i in range(height):
        for j in range(width):
            result[i][j] = array[i][j]

    return result

# print rounded array
def roundedArray(array, n):
    roundedArray = []
    for i in range(len(array)):
        try:
            roundedArray.append(round(array[i], n))
        except:
            roundedArray.append(array[i])
    return roundedArray

# Extract data array from file
def getDataFromFile(fn):
    result = []
    f = open(fn, 'r')
    fLines = f.readlines()
    fLen = len(fLines)
    f.close()

    # each line is a 1-dimensional array
    for i in range(fLen):

        # split each space by ' '
        thisLineSplit = fLines[i].split('\n')[0].split('\t')

        # convert into numeric value (except for '')
        for j in range(len(thisLineSplit)):
            if thisLineSplit[j] == '': thisLineSplit.pop() # except for ''
            else: thisLineSplit[j] = float(thisLineSplit[j])
            
        result.append(thisLineSplit) # append this line to the final result

    # return the final result
    return result

# return Neural Network
## layers
# I 60                   (keras.layers.InputLayer(input_shape=(60,)))
# FI                     (keras.layers.Flatten(input_shape=(len(trainI[0]),)))
# F                      (keras.layers.Flatten())
# D 16 relu              (keras.layers.Dense(16, activation='relu'))
# DO sigmoid             (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
# Drop 0.25              (keras.layers.Dropout(0.25))
# C2DI 32 3 3 12 12 relu (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
# C2D 32 3 3 relu        (keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# MP2D 2                 (keras.layers.MaxPooling2D(pool_size=2))
# R1 12                  (keras.layers.Reshape((12, 1), input_shape=(12,))
# R2 12 12               (keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))
def getNN(modelInfo, trainI, trainO):
    NN = []
    code = ''
    
    for i in range(len(modelInfo)):
        info = modelInfo[i].split('\n')[0]
        infoSplit = info.split(' ')

        # add layers to Neural Network as below
        if infoSplit[0] == 'I':
            NN.append(keras.layers.InputLayer(input_shape=(int(infoSplit[1]),)))
            code += 'NN.append(keras.layers.InputLayer(input_shape=(' + infoSplit[1] + ',)))\n'        
        elif info == 'FI':
            NN.append(keras.layers.Flatten(input_shape=(len(trainI[0]),)))
            code += 'NN.append(keras.layers.Flatten(input_shape=(' + str(len(trainI[0])) + ',)))\n'
        elif info == 'F':
            NN.append(keras.layers.Flatten())
            code += 'NN.append(keras.layers.Flatten())\n'
        elif infoSplit[0] == 'D':
            NN.append(keras.layers.Dense(int(infoSplit[1]), activation=infoSplit[2]))
            code += 'NN.append(keras.layers.Dense(' + infoSplit[1] + ', activation="' + infoSplit[2] + '"))\n'
        elif infoSplit[0] == 'DO':
            NN.append(keras.layers.Dense(len(trainO[0]), activation=infoSplit[1]))
            code += 'NN.append(keras.layers.Dense(' + str(len(trainO[0])) + ', activation="' + infoSplit[1] + '"))\n'
        elif infoSplit[0] == 'Drop':
            NN.append(keras.layers.Dropout(float(infoSplit[1])))
            code += 'NN.append(keras.layers.Dropout(' + infoSplit[1] + '))\n'
        elif infoSplit[0] == 'C1DI':
            NN.append(keras.layers.Conv1D(filters=int(infoSplit[1]), kernel_size=(int(infoSplit[2])), input_shape=(int(infoSplit[3]), 1), activation=infoSplit[4]))
            code += ('NN.append(keras.layers.Conv1D(filters=' + infoSplit[1] + ', kernel_size=(' + infoSplit[2] + '), input_shape=(' + infoSplit[3] + ', 1), '
                     + 'activation=' + infoSplit[4] + '))\n')
        elif infoSplit[0] == 'C1D':
            NN.append(keras.layers.Conv1D(filters=int(infoSplit[1]), kernel_size=(int(infoSplit[2])), activation=infoSplit[3]))
            code += 'NN.append(keras.layers.Conv1D(filters=' + infoSplit[1] + ', kernel_size=(' + infoSplit[2] + '), activation="' + infoSplit[3] + '"))\n'
        elif infoSplit[0] == 'MP1D':
            NN.append(keras.layers.MaxPooling1D(pool_size=int(infoSplit[1])))
        elif infoSplit[0] == 'C2DI':
            NN.append(keras.layers.Conv2D(filters=int(infoSplit[1]), kernel_size=(int(infoSplit[2]), int(infoSplit[3])),
                                          input_shape=(int(infoSplit[4]), int(infoSplit[5]), 1), activation=infoSplit[6]))
            code += ('NN.append(keras.layers.Conv2D(filters=' + infoSplit[1] + ', kernel_size=(' + infoSplit[2] + ', ' + infoSplit[3] + '), input_shape=('
                     + infoSplit[4] + ', ' + infoSplit[5] + ', 1), activation="' + infoSplit[6] + '"))\n')
        elif infoSplit[0] == 'C2D':
            NN.append(keras.layers.Conv2D(filters=int(infoSplit[1]), kernel_size=(int(infoSplit[2]), int(infoSplit[3])), activation=infoSplit[4]))
            code += 'NN.append(keras.layers.Conv2D(filters=' + infoSplit[1] + ', kernel_size=(' + infoSplit[2] + ', ' + infoSplit[3] + '), activation="' + infoSplit[4] + '"))\n'
        elif infoSplit[0] == 'MP2D':
            NN.append(keras.layers.MaxPooling2D(pool_size=int(infoSplit[1])))
            code += 'NN.append(keras.layers.MaxPooling2D(pool_size=' + infoSplit[1] + '))\n'
        elif infoSplit[0] == 'R1':
            NN.append(tf.keras.layers.Reshape((int(infoSplit[1]), 1), input_shape=(int(infoSplit[1]),)))
            code += 'NN.append(keras.layers.Reshape((' + infoSplit[1] + ', 1), input_shape=(' + infoSplit[1] + ',)))\n'
        elif infoSplit[0] == 'R2':
            NN.append(tf.keras.layers.Reshape((int(infoSplit[1]), int(infoSplit[2]), 1), input_shape=(int(infoSplit[1])*int(infoSplit[2]),)))
            code += 'NN.append(keras.layers.Reshape((' + infoSplit[1] + ', ' + infoSplit[2] + ', 1), input_shape=(' + str(int(infoSplit[1])*int(infoSplit[2])) + ',)))\n'

    # print code result
    print('\n <<< model code >>>')
    print(code)

    return NN

# modelInfo에 따라 optimizer 반환
## optimizers
# OP adadelta 0.001 0.95 1e-07    (tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07))
# OP adagrad 0.001 0.1 1e-07      (tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07))
# OP adam0 0.001                  (tf.keras.optimizers.Adam(0.001))
# OP adam1 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False))
# OP adamax 0.001 0.9 0.999 1e-07 (tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
# OP nadam 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
# OP rmsprop 0.001 0.9 0.0 1e-07  (tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07))
# OP sgd 0.01 0.0                 (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False))
def getOptimizer(modelInfo):
    op = None
    
    for i in range(len(modelInfo)):
        info = modelInfo[i].split('\n')[0]
        infoSplit = info.split(' ')

        # specify optimizer
        if infoSplit[0] == 'OP':
            if infoSplit[1] == 'adadelta': op = tf.keras.optimizers.Adadelta(learning_rate=float(infoSplit[2]),
                                                                             rho=float(infoSplit[3]),
                                                                             epsilon=float(infoSplit[4]))
            
            elif infoSplit[1] == 'adagrad': op = tf.keras.optimizers.Adagrad(learning_rate=float(infoSplit[2]),
                                                                             initial_accumulator_value=float(infoSplit[3]),
                                                                             epsilon=float(infoSplit[4]))
            
            elif infoSplit[1] == 'adam0': op = tf.keras.optimizers.Adam(float(infoSplit[2]))
            
            elif infoSplit[1] == 'adam1': op = tf.keras.optimizers.Adam(learning_rate=float(infoSplit[2]),
                                                                        beta_1=float(infoSplit[3]),
                                                                        beta_2=float(infoSplit[4]),
                                                                        epsilon=float(infoSplit[5]), amsgrad=False)
            
            elif infoSplit[1] == 'adamax': op = tf.keras.optimizers.Adamax(learning_rate=float(infoSplit[2]),
                                                                           beta_1=float(infoSplit[3]),
                                                                           beta_2=float(infoSplit[4]),
                                                                           epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'nadam': op = tf.keras.optimizers.Nadam(learning_rate=float(infoSplit[2]),
                                                                         beta_1=float(infoSplit[3]),
                                                                         beta_2=float(infoSplit[4]),
                                                                         epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'rmsprop': op = tf.keras.optimizers.RMSprop(learning_rate=float(infoSplit[2]),
                                                                             rho=float(infoSplit[3]),
                                                                             momentum=float(infoSplit[4]),
                                                                             epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'sgd': op = tf.keras.optimizers.SGD(learning_rate=float(infoSplit[2]),
                                                                     momentum=float(infoSplit[3]), nesterov=False)
            
            return op
