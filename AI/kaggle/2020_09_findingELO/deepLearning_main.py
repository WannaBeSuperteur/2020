import deepLearning_GPU
import deepLearning_GPU_helper as helper
import random
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from keras.models import Model, model_from_json
import readData as RD

# inputFileName      : train input data file
# outputFileName     : train output data file
# testFileName       : test input data file
# testOutputFileName : test output data file
# valid              : portion of validation data
#                      if >0, do not use test dataset and return validation input and output from input and output data
# normalizeTarget    : normalize target(output) value?
def deepLearning(inputFileName, outputFileName, testFileName, testOutputFileName, valid,
                 deviceName, epoch, printed, modelName, normalizeTarget):

    # read files
    # trainO : (originalOutput - meanOriginalOutput)/stdOriginalOutput
    trainI = helper.getDataFromFile(inputFileName, None) # input train data
    trainO = helper.getDataFromFile(outputFileName, None) # output train data (Sigmoid applied)
    testI = helper.getDataFromFile(testFileName, None) # test input data (set nullValue to 0)

    # apply sigmoid to train output data
    # trainO :   sigmoid(normalize(originalOutput))
    #          = sigmoid((originalOutput - meanOriginalOutput)/stdOriginalOutput)
    for i in range(len(trainO)):
        for j in range(len(trainO[0])):
            trainO[i][j] = helper.sigmoid(trainO[i][j])

    # for i in range(15): print(trainO[i])

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

    # read normalization info
    if normalizeTarget == True:
        fnorm = open('data_normalizeInfo.txt', 'r')
        fnormInfo = fnorm.readlines()
        fnormMean = float(fnormInfo[0].split(' ')[0]) # mean of training data
        fnormStd = float(fnormInfo[0].split(' ')[1]) # stddev of training data

    #### TEST when the value of valid is 0 ####
    if valid == 0:
        
        # NN and optimizer
        NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
        op = helper.getOptimizer(modelInfo) # optimizer

        #print(trainI[:5])
        #print(trainO[:5])

        try: # try reading test.h5 and test.json
            newModel = deepLearning_GPU.deepLearningModel(modelName, True)
            testOutput = deepLearning_GPU.modelOutput(newModel, testI)
        except: # do learning if test.h5 and test.json does not exist
            print('\n <<<< LEARNING >>>>\n')

            # False, True는 각각 dataPrint(학습데이터 출력 여부), modelPrint(model의 summary 출력 여부)
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, modelName, epoch, False, True, deviceName)

            newModel = deepLearning_GPU.deepLearningModel(modelName, True)
            testOutput = deepLearning_GPU.modelOutput(newModel, testI)

        # test
        print('\n <<<< TEST >>>>\n')

        # estimate
        outputLayer = testOutput[len(testOutput)-1]

        # inverse sigmoid
        # output:   denormalize(invSigmoid(sigmoid(normalize(originalOutput))))
        #         = denormalize((originalOutput - meanOriginalOutput)/stdOriginalOutput)
        #         = originalOutput
        for i in range(len(outputLayer)): # for each output data
            for j in range(len(outputLayer[0])): # for each value of output data
                outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])
                if normalizeTarget == True: outputLayer[i][j] = outputLayer[i][j] * fnormStd + fnormMean

        # write to file
        result = ''
        print('\n<<<< output layer >>>>')
        for i in range(len(outputLayer)): # for each output data
            for j in range(len(outputLayer[0])): # for each value of output data
                result += str(outputLayer[i][j]) + '\t'
            result += '\n'

        f = open(testOutputFileName.split('.')[0] + '_prediction.txt', 'w')
        f.write(result)
        f.close()

        # return final result
        finalResult = []
        for i in range(len(outputLayer)): # for each output data
            finalResult.append(outputLayer[i][0])

        return finalResult

    #### VALIDATION when the value of valid is >0 ####
    else:

        # make index-list of validation data
        inputSize = len(trainI)
        validSize = int(inputSize * valid)
        trainSize = inputSize - validSize

        validArray = []
        for i in range(inputSize): validArray.append(0)
        while sum(validArray) < validSize:
            validArray[random.randint(0, inputSize-1)] = 1

        # make train and validation data
        # _TrainO, _ValidO : sigmoid((originalOutput - meanOriginalOutput)/stdOriginalOutput)
        _TrainI = [] # training input
        _TrainO = [] # training output
        _ValidI = [] # valid input
        _ValidO = [] # valid output
        
        for i in range(inputSize):
            if validArray[i] == 0: # training data
                _TrainI.append(trainI[i])
                _TrainO.append(trainO[i])
            else: # validation data
                _ValidI.append(trainI[i])
                _ValidO.append(trainO[i])

        # model name for validation
        newModelName = modelName + 'Valid'

        # NN and optimizer
        NN = helper.getNN(modelInfo, _TrainI, _TrainO) # Neural Network
        op = helper.getOptimizer(modelInfo) # optimizer

        # output for validation        
        try: # try reading testValid.h5 and test.json
            validModel = deepLearning_GPU.deepLearningModel(newModelName, True)
            predictedValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)
        except: # do learning if testValid.h5 and test.json does not exist
            print('\n <<<< LEARNING >>>>\n')

            # False, True는 각각 dataPrint(학습데이터 출력 여부), modelPrint(model의 summary 출력 여부)
            # _TrainO : sigmoid((originalOutput - meanOriginalOutput)/stdOriginalOutput)
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', _TrainI, _TrainO, newModelName, epoch, False, True, deviceName)

            validModel = deepLearning_GPU.deepLearningModel(newModelName, True)
            predictedValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)

        # evaluation
        print('\n <<<< VALID >>>>\n')

        MAE = 0 # mean absolute error
        MSE = 0 # mean square error
        accuracy = 0 # accuracy

        # predicted validation output
        outputLayer = predictedValidO[len(predictedValidO)-1]
        
        # inverse sigmoid
        # output :   invSigmoid(sigmoid(normalize(originalOutput)))
        #          = (originalOutput - meanOriginalOutput)/stdOriginalOutput
        for i in range(len(outputLayer)): # for each output data
            for j in range(len(outputLayer[0])): # for each value of output data
                outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])
        
        # compute error
        # output  : denormalize((originalOutput - meanOriginalOutput)/stdOriginalOutput)
        #           = originalOutput
        # _Valid0 : denormalize(invSigmoid(sigmoid((originalOutput - meanOriginalOutput)/stdOriginalOutput)))
        #           = denormalize((originalOutput - meanOriginalOutput)/stdOriginalOutput)
        #           = originalOutput
        for i in range(len(outputLayer)): # for each output data
            for j in range(len(outputLayer[0])): # for each value of output data
                _ValidO[i][j] = helper.invSigmoid(_ValidO[i][j])
                if normalizeTarget == True:
                    _ValidO[i][j] = _ValidO[i][j] * fnormStd + fnormMean
                    outputLayer[i][j] = outputLayer[i][j] * fnormStd + fnormMean

        # compute error
        validCount = 0
        resultToWrite = ''
        for i in range(inputSize):
            if validArray[i] == 1:

                # compute errors and accuracy
                thisAE = abs(_ValidO[validCount][0] - outputLayer[validCount][0])
                thisSE = pow(_ValidO[validCount][0] - outputLayer[validCount][0], 2)
                MAE += thisAE
                MSE += thisSE
                if thisSE <= 0.5: accuracy += 1

                # print and write result
                newResultToWrite = ('[' + str(i) + '] pred = ' + str(int(outputLayer[validCount][0])) +
                                    ', real = ' + str(int(_ValidO[validCount][0])) +
                                    ', AE = ' + str(int(thisAE)) + ', SE = ' + str(int(thisSE)))
                resultToWrite += newResultToWrite + '\n'
                print(newResultToWrite)

                validCount += 1

        MAE /= validSize
        MSE /= validSize
        accuracy /= validSize

        # print evaluation result
        resultSummary = ''
        resultSummary += 'input size : ' + str(inputSize) + '\n'
        resultSummary += 'train size : ' + str(trainSize) + '\n'
        resultSummary += 'valid size : ' + str(validSize) + '\n'
        resultSummary += 'MAE        : ' + str(round(MAE, 6)) + '\n'
        resultSummary += 'MSE        : ' + str(round(MSE, 6)) + '\n'
        resultSummary += 'accuracy   : ' + str(round(accuracy, 6)) + '\n'
        resultSummary += 'pred avg   : ' + str(np.average(outputLayer, axis=0)) + '\n'
        resultSummary += 'real avg   : ' + str(np.average(_ValidO, axis=0)) + '\n'
        print(resultSummary)
        resultToWrite += resultSummary

        # write result file
        fvalid = open('data_valid_result.txt', 'w')
        fvalid.write(resultToWrite)
        fvalid.close()

# extract train input, train output, and test input from dataFrame and save them as a file
# dfTrain         : original dataFrame for training
# dfTest          : original dataFrame for test
# targetColName   : target column name for both training and test data
# exceptCols      : except for these columns for both training and test
# fn              : [train input file name, train output file name, test input file name, test output file name]
# normalizeTarget : normalize target(output) value?
def dataFromDF(dfTrain, dfTest, targetColName, exceptCols, fn, normalizeTarget):

    # make train input, train output, and test input dataFrame
    dfTrainInput = dfTrain.copy()
    dfTestInput = dfTest.copy()
    dfTrainOutput = dfTrain[targetColName]

    ### remove target column and except columns from training and test dataFrame ###
    colsToRemove = [targetColName] + exceptCols # columns to remove (do not use these columns)
    colsToRemoveIndexTrain = [] # indices for colsToRemove (training data)
    colsToRemoveIndexTest = [] # indices for colsToRemove (test data)

    print('\n *** columns to remove ***')
    print(colsToRemove)

    # get indices of columns to remove
    for i in range(len(dfTrainInput.columns)):
        if dfTrainInput.columns[i] in colsToRemove: colsToRemoveIndexTrain.append(i)
    for i in range(len(dfTestInput.columns)):
        if dfTestInput.columns[i] in colsToRemove: colsToRemoveIndexTest.append(i)

    print('\n *** columns to remove (indices) ***')
    print('train : ' + str(colsToRemoveIndexTrain))
    print('test  : ' + str(colsToRemoveIndexTest))

    # change into np.array
    dfTrainInput = np.array(dfTrainInput)
    dfTestInput = np.array(dfTestInput)
    
    # remove columns
    dfTrainInput = np.delete(dfTrainInput, colsToRemoveIndexTrain, 1)
    dfTestInput = np.delete(dfTestInput, colsToRemoveIndexTest, 0)

    # print each dataFrame
    print('\n *** dataFrame (training input) ***')
    print(dfTrainInput)

    print('\n *** dataFrame (test input) ***')
    print(dfTestInput)

    print('\n *** dataFrame (training output) ***')
    print(dfTrainOutput)
    
    # make train input, train output, and test input array
    dfTrainInputArray = np.array(dfTrainInput)
    dfTrainOutputArray = np.array(dfTrainOutput)
    dfTestInputArray = np.array(dfTestInput)

    # normalize training output data
    if normalizeTarget == True:
        dfTrainOutputMean = np.mean(dfTrainOutputArray) # mean of dfTrainOutputArray
        dfTrainOutputStddev = np.std(dfTrainOutputArray) # stddev of dfTrainOutputArray
        
        for i in range(len(dfTrainOutputArray)):
            dfTrainOutputArray[i] = (dfTrainOutputArray[i] - dfTrainOutputMean)/dfTrainOutputStddev

    # train output array -> [a, b, c, ...] to [[a], [b], [c], ...]
    dfTrainOutputArray_ = []
    for i in range(len(dfTrainOutputArray)): dfTrainOutputArray_.append([dfTrainOutputArray[i]])

    # make train input, train output, and test output file
    RD.saveArray(fn[0], dfTrainInputArray)
    RD.saveArray(fn[1], dfTrainOutputArray_)
    RD.saveArray(fn[2], dfTestInputArray)

    if normalizeTarget == True:
        fnorm = open('data_normalizeInfo.txt', 'w')
        fnorm.write(str(dfTrainOutputMean) + ' ' + str(dfTrainOutputStddev) + ' ')
        fnorm.close()

# deep learning procedure
# valid = 0 : training -> test
# valid > 0 : training -> validation (vaildation data rate = valid)
# return final result (None if valid > 0)
def deepLearningProcedure(fn, valid, normalizeTarget):

    # deep learning configuration
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # training and return result
    if valid == 0:
        finalResult = deepLearning(fn[0], fn[1], fn[2], fn[3], 0, deviceName, epoch, printed, 'test', normalizeTarget)
        return finalResult
    else:
        deepLearning(fn[0], fn[1], fn[2], fn[3], valid, deviceName, epoch, printed, 'test', normalizeTarget)
        return None
