import deepLearning_GPU
import deepLearning_GPU_helper as helper
import random
import tensorflow as tf
import numpy as np
import math
import randomData
from tensorflow import keras
from keras.models import Model, model_from_json

# inputFileName      : train input data file (line 1)
# outputFileName     : train output data file (line 2)
# testFileName       : test input data file (line 3)
# testOutputFileName : test output data file (line 4)
# imgHeight          : size(width=height) of image (line 5)
def deepLearning(inputFileName, outputFileName, testFileName, testOutputFileName, imgHeight, deviceName, epoch, printed, modelName):

    # read files
    trainI = helper.getDataFromFile(inputFileName, imgHeight) # input train data
    trainO = helper.getDataFromFile(outputFileName, None) # output train data (Sigmoid applied)
    testI = helper.getDataFromFile(testFileName, imgHeight) # test input data (set nullValue to 0)

    # apply sigmoid to train output data
    for i in range(len(trainO)):
        for j in range(len(trainO[0])):
            trainO[i][j] = helper.sigmoid(trainO[i][j])

    # flatten trainI: (N, size, size) -> (N, size*size)
    for i in range(len(trainI)): trainI[i] = helper.flatten(trainI[i])

    print('')
    print(' ---- number of rows ----')
    print('input  size: ' + str(len(trainI)))
    print('output size: ' + str(len(trainO)))
    print('test   size: ' + str(len(testI)))
    print('')

    # print input, output, and test data
    if printed != 0:
        print('\n ---- original input data ----\n')
        for i in range(len(trainI)): print(helper.roundedArray(trainI[i], 6))

        print('\n ---- original output data ----\n')
        for i in range(len(trainO)): print(helper.roundedArray(trainO[i], 6))

        print('\n ---- original test data ----\n')
        for i in range(len(testI)): print(helper.roundedArray(testI[i], 6))
        
    # model design using deepLearning_model.txt, in the form of
    # activation function of final layer is always 'sigmoid'
    f = open('deepLearning_model.txt', 'r')
    modelInfo = f.readlines()
    f.close()

    # NN and optimizer
    NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
    op = helper.getOptimizer(modelInfo) # optimizer

    try: # try reading test.h5 and test.json
        newModel = deepLearning_GPU.deepLearningModel(modelName, True)
        testOutput = deepLearning_GPU.modelOutput(newModel, testI)
    except: # do learning if test.h5 and test.json does not exist
        print('\n <<<< LEARNING >>>>\n')

        # False, True는 각각 dataPrint(학습데이터 출력 여부), modelPrint(model의 summary 출력 여부)
        print(trainO[0])
        deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, modelName, epoch, False, True, deviceName)

        newModel = deepLearning_GPU.deepLearningModel(modelName, True)
        testOutput = deepLearning_GPU.modelOutput(newModel, testI)

    # test
    print('\n <<<< TEST >>>>\n')

    # estimate
    outputLayer = testOutput[len(testOutput)-1]

    # inverse sigmoid
    for i in range(len(outputLayer)): # for each output data
        for j in range(len(outputLayer[0])): # for each value of output data
            outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])

    # write to file
    result = ''
    print('\n<<<< output layer >>>>')
    for i in range(len(outputLayer)): # for each output data
        for j in range(len(outputLayer[0])): # for each value of output data
            result += str(outputLayer[i][j]) + '\t'
        result += '\n'
    print(result)

    f = open(testOutputFileName.split('.')[0] + '_prediction.txt', 'w')
    f.write(result)
    f.close()
