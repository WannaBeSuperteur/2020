import deepLearning_GPU
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Model, model_from_json
import gc

global ALLTEXT
ALLTEXT = ''

def getDataFromFile(fn, splitter):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    result = []
    for i in range(len(flines)):
        row = flines[i].split('\n')[0].split(splitter)
        for j in range(len(row)): row[j] = float(row[j])
        result.append(row)

    return result

def sign(val):
    if val > 0: return 1
    elif val == 0: return 0
    else: return -1

def trainAndTest(lr, dropout):
    global ALLTEXT
    
    # training and test inputs and outputs
    trainI = []
    trainO = []
    testI = []
    testO = []
    
    for i in range(len(inputs)):
        if random.random() < 0.2:
            testI.append(inputs[i])
            testO.append(outputs[i])
        else:
            trainI.append(inputs[i])
            trainO.append(outputs[i])

    # model design
    NN = [tf.keras.layers.Flatten(input_shape=(len(inputs[0]),)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(len(outputs[0]), activation='sigmoid')]
    op = tf.keras.optimizers.Adam(lr)

    # learning
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, 'test', 50, False, True, deviceName)

    # test
    newModel = deepLearning_GPU.deepLearningModel('test', True)
    testOutput = deepLearning_GPU.modelOutput(newModel, testI)

    # estimate
    outputLayer = testOutput[len(testOutput)-1]
    avgSquareError = 0.0
    
    for i in range(len(testO)):
        op = outputLayer[i] # output layer output
        to = testO[i] # test output
        squareError = pow(op[0]-to[0], 2) + pow(op[1]-to[1], 2)
        avgSquareError += squareError
    avgSquareError /= len(testO)

    toAppend = str(round(lr, 8)) + ' ' + str(round(dropout, 8)) + ' ' + str(round(avgSquareError, 8))
    ALLTEXT += toAppend + '\n'
    print(toAppend)
    
    return avgSquareError

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')

    inputs = getDataFromFile('inputs.txt', '\t')
    outputs = getDataFromFile('outputs.txt', '\t')

    # set target space
    # learning rate between 0.0 and 0.25
    lrMin = 0.0
    lrMax = 0.25

    # dropout between 0.0 and 0.75
    dropoutMin = 0.0
    dropoutMax = 0.75

    np.set_printoptions(precision=4, linewidth=150)

    # grid search
    if True:
        # grid search
        toAppend0 = '\n< grid search >'
        toAppend1 = 'LR: ' + str(lrMin) + '~' + str(lrMax) + ', DROPOUT: ' + str(dropoutMin) + '~' + str(dropoutMax)
        toAppend2 = 'lr drouput avgSquareError'
        
        ALLTEXT += toAppend0 + '\n' + toAppend1 + '\n' + toAppend2 + '\n'
        print(toAppend0)
        print(toAppend1)
        print(toAppend2)
        
        grids = []
        for i in range(11):
            for j in range(11):
                lr = i * 0.1 * (lrMax-lrMin) + lrMin
                dropout = j * 0.1 * (dropoutMax-dropoutMin) + dropoutMin
                grids.append([lr, dropout, trainAndTest(lr, dropout)])
                gc.collect()

        # find minError for the result
        minError = 1000000000.0
        for i in range(121):
            if grids[i][2] < minError:
                minError = grids[i][2]
                minErrorLr = grids[i][0]
                minErrorDropout = grids[i][1]

        # set target space
        toAppend = '\n< set target space >'
        ALLTEXT += toAppend + '\n'
        print(toAppend)
        
        lrGap = lrMax-lrMin
        dropoutGap = dropoutMax-dropoutMin

        toAppend0 = 'lrMin=' + str(lrMin) + ' lrMax=' + str(lrMax) + ' lrGap=' + str(lrGap)
        toAppend1 = 'dropoutMin=' + str(dropoutMin) + ' dropoutMax=' + str(dropoutMax) + ' dropoutGap=' + str(dropoutGap)
        toAppend2 = 'minErrorLr=' + str(minErrorLr) + ' minErrorDropout=' + str(minErrorDropout)
        toAppend3 = 'new lrMin=' + str(minErrorLr - lrGap * 0.2) + ' new lrMax=' + str(minErrorLr + lrGap * 0.2)
        toAppend4 = 'new dropoutMin=' + str(minErrorDropout - dropoutGap * 0.2) + ' new dropoutMax=' + str(minErrorDropout + dropoutGap * 0.2)

        ALLTEXT += toAppend0 + '\n' + toAppend1 + '\n' + toAppend2 + '\n' + toAppend3 + '\n' + toAppend4 + '\n'
        print(toAppend0)
        print(toAppend1)
        print(toAppend2)
        print(toAppend3)
        print(toAppend4)
        
        lrMin = max(0, minErrorLr - lrGap * 0.2)
        lrMax = min(1, minErrorLr + lrGap * 0.2)
        dropoutMin = max(0, minErrorDropout - dropoutGap * 0.2)
        dropoutMax = min(1, minErrorDropout + dropoutGap * 0.2)

        # write to file
        f = open('result_grid_only.txt', 'w')
        f.write(ALLTEXT)
        f.close()
