import deepLearning_GPU
import deepLearning_GPU_helper as helper
import random
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from keras.models import Model, model_from_json

# save result array
def saveArray(fn, _2dArray):
    
    result = ''
    rows = len(_2dArray) # rows of 2d array
    cols = len(_2dArray[0]) # cols of 2d array

    # append to result
    for i in range(rows):
        for j in range(cols):
            if j < cols-1: result += str(_2dArray[i][j]) + '\t'
            else: result += str(_2dArray[i][j])

        result += '\n'

    # write to file
    f = open(fn, 'w')
    f.write(result)
    f.close()

# inputFileName      : train input data file name
# outputFileName     : train output data file name
# testFileName       : test input data file name
# testOutputFileName : test output (prediction) data file name
# testOutputReal     : test output (real) data file name (if not None, compare output prediction with real test output data)
# test_report        : test report file name
# validRate          : rate of validation data among training data (0 -> use test data)
# valid_report       : vaildation report file name
# modelConfig        : model configuration file name
def deepLearning(inputFileName, outputFileName, testFileName, testOutputFileName, testOutputReal,
                 test_report, validRate, valid_report, modelConfig, deviceName, epoch, printed, modelName):

    ##############################
    ##                          ##
    ##       0. READ DATA       ##
    ##                          ##
    ##############################
    
    # read files
    print('[00] reading files...')
    trainI = helper.getDataFromFile(inputFileName) # input train data
    trainO = helper.getDataFromFile(outputFileName) # output train data (Sigmoid applied)
    testI = helper.getDataFromFile(testFileName) # test input data (set nullValue to 0)

    # apply sigmoid to train output data
    for i in range(len(trainO)):
        for j in range(len(trainO[0])):
            trainO[i][j] = helper.sigmoid(trainO[i][j])

    # print input, output, and test data
    if printed != 0:
        print('\n ---- original input data (' + str(len(trainI)) + ') ----\n')
        for i in range(len(trainI)): print(helper.roundedArray(trainI[i], 6))

        print('\n ---- original output data (' + str(len(trainO)) + ') ----\n')
        for i in range(len(trainO)): print(helper.roundedArray(trainO[i], 6))

        print('\n ---- original test data (' + str(len(testI)) + ') ----\n')
        for i in range(len(testI)): print(helper.roundedArray(testI[i], 6))

    ##############################
    ##                          ##
    ##   1. READ MODEL CONFIG   ##
    ##                          ##
    ##############################
        
    # model design using model configuration file
    # activation function of final layer is always 'sigmoid'
    print('[10] reading model configuration...')
    f = open(modelConfig, 'r')
    modelInfo = f.readlines()
    f.close()

    ##############################
    ##                          ##
    ##   2A. TRAINING / TEST    ##
    ##                          ##
    ##############################

    # if the model already exists, input the test input to the NN and get the result
    # if the model does not exist, newly train NN using training input and output data and then do testing procedure
    if validRate == 0:

        # NN and optimizer
        print('[11] obtaining neural network and optimizer info...')
        NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
        op = helper.getOptimizer(modelInfo) # optimizer
        
        try: # try reading test.h5 and test.json
            print('[20] reading model [ ' + modelName + ' ]...')
            newModel = deepLearning_GPU.deepLearningModel(modelName, True)
            testO = deepLearning_GPU.modelOutput(newModel, testI)
            
        except: # do learning if test.h5 and test.json does not exist
            print('[21] learning...')

            # False, True는 각각 dataPrint(학습데이터 출력 여부), modelPrint(model의 summary 출력 여부)
            print(trainO[0])
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, modelName, epoch, False, True, deviceName)

            print('[22] reading learned model [ ' + modelName + ' ]...')
            newModel = deepLearning_GPU.deepLearningModel(modelName, True)
            testO = deepLearning_GPU.modelOutput(newModel, testI)

        # test
        print('[23] testing...')

        # estimate
        OL = testO[len(testO)-1]

        # inverse sigmoid
        for i in range(len(OL)): # for each output data
            for j in range(len(OL[0])): # for each value of output data
                OL[i][j] = helper.invSigmoid(OL[i][j])

        # write to file
        print('[24] writing test result to file [ ' + testOutputFileName + ' ]...')
        result = ''
        for i in range(len(OL)): # for each output data
            for j in range(len(OL[0])): # for each value of output data
                result += str(OL[i][j]) + '\t'
            result += '\n'

        f = open(testOutputFileName, 'w')
        f.write(result)
        f.close()

        ##############################
        ##                          ##
        ##  2A+. WRITE TEST REPORT  ##
        ##                          ##
        ##############################

        # compare prediction output data with real output data and write report
        if testOutputReal != None:
            print('[25] writing report for test result, to file [ ' + test_report + ' ]...')
            
            MAE = 0 # mean absolute error
            MSE = 0 # mean square error
            accuracy = 0 # accuracy
        
            testO_pred = helper.getDataFromFile(testOutputFileName) # prediction of test
            testO_real = helper.getDataFromFile(testOutputReal) # real value

            # assertion for length of data
            assert(len(testO_pred) == len(testO_real))
            assert(len(testO_pred[0]) == len(testO_real[0]))

            testSize = len(testO_pred) # number of test data
            testCols = len(testO_pred[0]) # number of columns in each test data
            report = 'predict\treal\t' * testCols + '\n' # content of the report file

            for i in range(testSize):
                for j in range(testCols):

                    # compute errors and accuracy
                    MAE += abs(testO_pred[i][j] - testO_real[i][j])
                    MSE += pow(testO_pred[i][j] - testO_real[i][j], 2)

                    report += str(testO_pred[i][j]) + '\t' + str(testO_real[i][j]) + '\t'

                # compute accuracy
                if helper.argmax(testO_pred[i]) == helper.argmax(testO_real[i]): accuracy += 1
                
                report += '\n'

            # get the average of MAE, MSE and accuracy
            MAE /= (testSize * testCols)
            MSE /= (testSize * testCols)
            accuracy /= testSize

            # write report file
            report = ('MAE=' + str(MAE) + ' MSE=' + str(MSE) + ' accuracy=' + str(accuracy) + '\n') + report
            
            f = open(test_report, 'w')
            f.write(report)
            f.close()

            # return final result (-1 means not used)
            return (MAE, MSE, accuracy, -1, -1)

    ##############################
    ##                          ##
    ##      2B. VALIDATION      ##
    ##                          ##
    ##############################

    # validation (if validation rate > 0)
    else:

        ##############################
        ##                          ##
        ##   2B-0. DATA TO VALID    ##
        ##                          ##
        ##############################
        
        # make index-list of validation data
        print('[26] deciding data to validate...')
        inputSize = len(trainI)
        validSize = int(inputSize * validRate)
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

        ##############################
        ##                          ##
        ## 2B-1. TRAIN (MAKE MODEL) ##
        ##                          ##
        ##############################

        # model name for validation
        newModelName = modelName + 'Valid'
        print('[27] training [ ' + newModelName + ' ]...')

        # NN and optimizer
        NN = helper.getNN(modelInfo, _TrainI, _TrainO) # Neural Network
        op = helper.getOptimizer(modelInfo) # optimizer

        # output for validation        
        try: # try reading the validation model
            validModel = deepLearning_GPU.deepLearningModel(newModelName, True)
            predValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)
        except: # do learning if the validation model does not exist
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', _TrainI, _TrainO, newModelName, epoch, False, True, deviceName)
            validModel = deepLearning_GPU.deepLearningModel(newModelName, True)
            predValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)

        ##############################
        ##                          ##
        ##     2B-2. VALIDATION     ##
        ##                          ##
        ##############################
        print('[28] validating and writing result [ ' + valid_report + ' ]...')
        
        MAE = 0 # mean absolute error
        MSE = 0 # mean square error
        accuracy = 0 # accuracy

        # predicted validation output
        _ValidOP = predValidO[len(predValidO)-1]
        
        # inverse sigmoid for PREDICTED validation output
        for i in range(len(_ValidOP)): # for each output data
            for j in range(len(_ValidOP[0])): # for each value of output data
                _ValidOP[i][j] = helper.invSigmoid(_ValidOP[i][j])
        
        # inverse sigmoid for REAL validation output
        for i in range(len(_ValidO)): # for each output data
            for j in range(len(_ValidO[0])): # for each value of output data
                _ValidO[i][j] = helper.invSigmoid(_ValidO[i][j])
        
        # compute error
        validCount = 0
        resultToWrite = ''
        outputCols = len(_ValidO[0])

        # for each data
        for i in range(inputSize):

            # validation for data whose value of valid array is 1
            if validArray[i] == 1:

                # compute MAE and MSE
                for j in range(outputCols):
                    MAE += abs(_ValidO[validCount][0] - _ValidOP[validCount][0])
                    MSE += pow(_ValidO[validCount][0] - _ValidOP[validCount][0], 2)

                # compute accuracy
                if helper.argmax(_ValidO[validCount]) == helper.argmax(_ValidOP[validCount]): accuracy += 1

                # print and write result
                newResultToWrite = ('[' + str(i) + '] pred = '
                                    + str(np.round_(_ValidOP[validCount], 6)) + ', real = ' + str(np.round_(_ValidO[validCount], 6)))
                resultToWrite += newResultToWrite + '\n'

                validCount += 1

        # get the average of MAE, MSE and accuracy
        MAE /= (validSize * outputCols)
        MSE /= (validSize * outputCols)
        accuracy /= validSize

        # print evaluation result
        resultSummary = ''
        resultSummary += 'input size : ' + str(inputSize) + '\n'
        resultSummary += 'train size : ' + str(trainSize) + '\n'
        resultSummary += 'valid size : ' + str(validSize) + '\n'
        resultSummary += 'MAE        : ' + str(round(MAE, 6)) + '\n'
        resultSummary += 'MSE        : ' + str(round(MSE, 6)) + '\n'
        resultSummary += 'accuracy   : ' + str(round(accuracy, 6)) + '\n'
        resultSummary += 'pred avg   : ' + str(np.average(_ValidOP, axis=0)) + '\n'
        resultSummary += 'real avg   : ' + str(np.average(_ValidO, axis=0)) + '\n'
        print(resultSummary)
        resultToWrite += resultSummary

        # write result file
        fvalid = open(valid_report, 'w')
        fvalid.write(resultToWrite)
        fvalid.close()

        # return final result
        return (MAE, MSE, accuracy, np.average(_ValidOP, axis=0), np.average(_ValidO, axis=0))

# extract train input, train output, and test input from dataFrame and save them as a file
# dfTrain         : original dataFrame for training
# dfTest          : original dataFrame for test
# targetColName   : target column name for both training and test data
# exceptCols      : except for these columns for both training and test
# fn              : [train input file name, train output file name, test input file name, test output file name]
# normalizeTarget : normalize target(output) value?
def dataFromDF(dfTrain, dfTest, targetColName, exceptCols, normalizeTarget):

    # read configuration file
    f = open('config.txt', 'r')
    fl = f.readlines()
    f.close()
    for i in range(len(fl)): fl[i] = fl[i].split('\n')[0]

    # init values as None
    fn = [None, None, None]

    # extract configuration
    # trainInput     : train input data file name
    # trainOutput    : train output data file name
    # testInput      : test input data file name
    for i in range(len(fl)):
        configSplit = fl[i].split(' ') # split

        if configSplit[0] == 'trainInput': fn[0] = configSplit[1]
        elif configSplit[0] == 'trainOutput': fn[1] = configSplit[1]
        elif configSplit[0] == 'testInput': fn[2] = configSplit[1]

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

    print('\n *** dataFrame (training output - not normalized) ***')
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

    print('\n *** dataFrame (training output - normalized) ***')
    for i in range(5): print(dfTrainOutputArray[i])
    print('...')
    for i in range(len(dfTrainOutputArray)-5, len(dfTrainOutputArray)): print(dfTrainOutputArray[i])
    print('')

    # train output array -> [a, b, c, ...] to [[a], [b], [c], ...]
    dfTrainOutputArray_ = []
    for i in range(len(dfTrainOutputArray)): dfTrainOutputArray_.append([dfTrainOutputArray[i]])

    # make train input, train output, and test output file
    saveArray(fn[0], dfTrainInputArray)
    saveArray(fn[1], dfTrainOutputArray_)
    saveArray(fn[2], dfTestInputArray)

    if normalizeTarget == True:
        fnorm = open('data_normalizeInfo.txt', 'w')
        fnorm.write(str(dfTrainOutputMean) + ' ' + str(dfTrainOutputStddev) + ' ')
        fnorm.close()
