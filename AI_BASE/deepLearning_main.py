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

# denormalize if normalized info is available
# normalizeName     : normalize info (average and stddev) file name
# testSize          : number of records (=number of rows) of test data
# testCols          : number of columns of test data
# testO_pred        : prediction of test output
# trainOutputAvg    : average of training output (each column, so array form)
# trainOutputStddev : stddev of training output (each column, so array form)
def denormalize(normalizeName, testSize, testCols, testO_pred, trainOutputAvg, trainOutputStddev):
    
    if normalizeName != None:
        print('[26] denormalizing prediction output and real value...')

        for i in range(testSize):
            for j in range(testCols):
                testO_pred[i][j] = testO_pred[i][j] * trainOutputStddev[j] + trainOutputAvg[j]
    else:
        print('[27] Do not denormalizing prediction output and real value because avg and std of them are not available.')

# write normalize info (average and stddev) of train output array
# trainOutputArray : train output array
# normalizeName    : normalize info (average and stddev) file name
def writeNormalizeInfo(trainOutputArray, normalizeName):

    normalizeInfo = ''
    
    for i in range(len(trainOutputArray[0])):

        # write normalize info
        trainOutputMean = np.mean(trainOutputArray, axis=0)[i] # mean of dfTrainOutputArray
        trainOutputStddev = np.std(trainOutputArray, axis=0)[i] # stddev of dfTrainOutputArray
        normalizeInfo += str(trainOutputMean) + ' ' + str(trainOutputStddev) + '\n'

        # normalize output data
        for j in range(len(trainOutputArray)):
            trainOutputArray[j][i] = (trainOutputArray[j][i] - trainOutputMean)/trainOutputStddev

    # write average and stddev of train output column
    fnorm = open(normalizeName, 'w')
    fnorm.write(normalizeInfo)
    fnorm.close()

# write test result
# test_report        : test report file name
# testOutputFileName : test output (prediction) data file name
# testOutputReal     : test output (real) data file name (if not None, compare output prediction with real test output data)
# normalizeName      : normalize info (average and stddev) file name
# trainOutputAvg     : average of training output (each column, so array form)
# trainOutputStddev  : stddev of training output (each column, so array form)
def writeTestResult(test_report, testOutputFileName, testOutputReal, normalizeName, trainOutputAvg, trainOutputStddev):
    print('[25] writing report for test result, to file [ ' + test_report + ' ]...')
            
    MAE = 0 # mean absolute error
    MSE = 0 # mean square error
    accuracy = 0 # accuracy
        
    testO_pred = helper.getDataFromFile(testOutputFileName) # prediction of test
    testO_real = helper.getDataFromFile(testOutputReal) # real value

    testSize = len(testO_pred) # number of test data
    testCols = len(testO_pred[0]) # number of columns in each test data
    report = 'predict\treal\t' * testCols + '\n' # content of the report file

    # denormalize if normalized info is available
    denormalize(normalizeName, testSize, testCols, testO_pred, trainOutputAvg, trainOutputStddev)

    # assertion for length of data
    try:
        assert(len(testO_pred) == len(testO_real))
        assert(len(testO_pred[0]) == len(testO_real[0]))
    except:
        print(' **** the number of rows or columns in PREDICTED test output file ( ' + testOutputFileName +
              ' ) and REAL test output file ( ' + testOutputReal + ' ) are not the same. ****')
        return

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

# inputFileName      : train input data file name
# outputFileName     : train output data file name
# testFileName       : test input data file name (or test input array)
# testOutputFileName : test output (prediction) data file name
# testOutputReal     : test output (real) data file name (if not None, compare output prediction with real test output data)
# test_report        : test report file name
# validRate          : rate of validation data among training data (0 -> use test data)
# valid_report       : vaildation report file name
# modelConfig        : model configuration file name
def deepLearning(inputFileName, outputFileName, testFileName, testOutputFileName, testOutputReal,
                 test_report, validRate, valid_report, modelConfig, deviceName, epoch, printed, modelName):

    # You can do only 'training' or 'testing' by setting some arguments as None.
    # inputFileName == None and outputFileName == None -> testing only
    # testFileName == None                             -> training only
    # for validation, you can set testFileName == None <- validation uses training data only

    ##############################
    ##                          ##
    ##       0. READ DATA       ##
    ##                          ##
    ##############################
    
    # read files
    print('[00] reading train input / train output / test input files...')
    
    trainI = None
    trainO = None
    testI = None

    # input train data
    if inputFileName != None: trainI = helper.getDataFromFile(inputFileName)

    #  output train data (Sigmoid applied)
    if outputFileName != None: trainO = helper.getDataFromFile(outputFileName)

    # test input data (set nullValue to 0)
    # set testI (array) as testFileName, if testFileName is an array
    if isinstance(testFileName, list):
        testI = testFileName

    # set testI (array) as test data from the file named as testFileName
    else:
        if testFileName != None: testI = helper.getDataFromFile(testFileName)

    # read configuration file (to get normalization info)
    print('[01] reading configuration files...')
    f = open('config.txt', 'r')
    fl = f.readlines()
    f.close()
    for i in range(len(fl)): fl[i] = fl[i].split('\n')[0]

    normalizeName = None
    validInterval = 1

    # extract configuration
    # trainInput     : train input data file name
    # trainOutput    : train output data file name
    # testInput      : test input data file name
    for i in range(len(fl)):
        configSplit = fl[i].split('\n')[0].split(' ') # split

        # normalize info file name
        if configSplit[0] == 'normalizeName':
            normalizeName = configSplit[1]
            if normalizeName == 'None': normalizeName = None

        # validation interval
        elif configSplit[0] == 'validInterval':
            validInterval = int(configSplit[1])

    # read normalization info file
    if normalizeName != None and trainO != None:
        print('[02] calculating and writing average and stddev...')
        
        trainOutputAvg = np.mean(trainO, axis=0) # average of train output value
        trainOutputStddev = np.std(trainO, axis=0) # stddev of train output value

        # normalize training output data and write avg and stddev
        writeNormalizeInfo(trainO, normalizeName)
    else:
        print('[03] Reading average and stddev failed.')
        trainOutputAvg = None
        trainOutputStddev = None

    # apply sigmoid to train output data
    if trainO != None:
        print('[04] applying sigmoid to train output data...')
        for i in range(len(trainO)):
            for j in range(len(trainO[0])):
                trainO[i][j] = helper.sigmoid(trainO[i][j])

    # print input, output, and test data
    if printed != 0:
        if trainI != None:
            print('\n ---- original input data (' + str(len(trainI)) + ') ----\n')
            for i in range(len(trainI)): print(helper.roundedArray(trainI[i], 6))

        if trainO != None:
            print('\n ---- original output data (' + str(len(trainO)) + ') ----\n')
            for i in range(len(trainO)): print(helper.roundedArray(trainO[i], 6))

        if testI != None:
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

        if trainI != None and trainO != None:
            NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
            op = helper.getOptimizer(modelInfo) # optimizer
            loss = helper.getLoss(modelInfo) # loss
        
        try: # try reading test.h5 and test.json
            print('[20] reading model [ ' + modelName + ' ]...')
            newModel = deepLearning_GPU.deepLearningModel(modelName, op, loss, True)
            testO = deepLearning_GPU.modelOutput(newModel, testI)
            
        except: # do learning if test.h5 and test.json does not exist
            print('[21] learning...')

            # False, True는 각각 dataPrint(학습데이터 출력 여부), modelPrint(model의 summary 출력 여부)
            print(trainO[0])
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, modelName, epoch, False, True, deviceName)

            print('[22] reading learned model [ ' + modelName + ' ]...')
            newModel = deepLearning_GPU.deepLearningModel(modelName, op, loss, True)

            # get test output if testI is not None
            if testI == None:
                print('test input file name (testInput) is None.')
                return
            else:
                testO = deepLearning_GPU.modelOutput(newModel, testI)

        # test
        print('[23] testing...')

        # estimate
        OL = testO[len(testO)-1]

        # inverse sigmoid
        for i in range(len(OL)): # for each output data
            for j in range(len(OL[0])): # for each value of output data
                OL[i][j] = helper.invSigmoid(OL[i][j])

        # check if test output exists, before writing test output file
        try:
            test = open(testOutputFileName, 'r')
            test.close()
            print(' **** Delete test output file (' + testOutputFileName + ') first. ****')
            return
        except:
            pass

        # write to file
        print('[24] writing test result to file [ ' + testOutputFileName + ' ]...')

        # open file
        f = open(testOutputFileName, 'a')

        result = ''
        for i in range(len(OL)): # for each output data
            if i % 1000 == 0: print(str(i) + ' / ' + str(len(OL)))
            
            for j in range(len(OL[0])): # for each value of output data
                result += str(OL[i][j]) + '\t'
            result += '\n'

            # flush every 10,000 steps
            if i % 10000 == 0:
                f.write(result)
                result = ''

        # final append
        f.write(result)
        f.close()

        ##############################
        ##                          ##
        ##  2A+. WRITE TEST REPORT  ##
        ##                          ##
        ##############################

        # compare prediction output data with real output data and write report
        if testOutputReal != None:
            try:
                writeTestResult(test_report, testOutputFileName, testOutputReal, normalizeName, trainOutputAvg, trainOutputStddev)
            except:
                pass

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
        print('[28] deciding data to validate...')
        inputSize = len(trainI)
        validSize = int(inputSize * validRate)
        trainSize = inputSize - validSize

        validArray = []
        for i in range(inputSize): validArray.append(0)
        while sum(validArray) < validSize:

            # start index for validation
            validStartIndex = int(random.randint(0, inputSize-1) / validInterval) * validInterval

            # set data[validStartIndex : validStartIndex + validInterval] as validation data
            for i in range(validStartIndex, validStartIndex + validInterval): validArray[i] = 1

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
        print('[29] training [ ' + newModelName + ' ]...')

        # NN and optimizer
        NN = helper.getNN(modelInfo, _TrainI, _TrainO) # Neural Network
        op = helper.getOptimizer(modelInfo) # optimizer
        loss = helper.getLoss(modelInfo) # loss

        # output for validation        
        try: # try reading the validation model
            validModel = deepLearning_GPU.deepLearningModel(newModelName, op, loss, True)
            predValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)
        except: # do learning if the validation model does not exist
            deepLearning_GPU.deepLearning(NN, op, loss, _TrainI, _TrainO, newModelName, epoch, False, True, deviceName)
            validModel = deepLearning_GPU.deepLearningModel(newModelName, op, loss, True)
            predValidO = deepLearning_GPU.modelOutput(validModel, _ValidI)

        ##############################
        ##                          ##
        ##     2B-2. VALIDATION     ##
        ##                          ##
        ##############################
        print('[30] validating and writing result [ ' + valid_report + ' ]...')
        
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

        # denormalize if normalized info is available (denormalize whole trainO)
        denormalize(normalizeName, len(_ValidOP), len(_ValidOP[0]), _ValidOP, trainOutputAvg, trainOutputStddev)
        denormalize(normalizeName, len(_ValidO), len(_ValidO[0]), _ValidO, trainOutputAvg, trainOutputStddev)
        
        # compute error
        validCount = 0
        resultToWrite = ''
        outputCols = len(_ValidO[0])

        # for each data

        # set edgeitems and linewidth as infinite
        np.set_printoptions(edgeitems=10000, linewidth=1000000)
        
        for i in range(inputSize):
            if i % 1000 == 0: print(str(i) + ' / ' + str(inputSize))

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

        # recover edgeitems and linewidth
        np.set_printoptions(edgeitems=10000, linewidth=1000000)

        # get the average of MAE, MSE and accuracy
        MAE /= (validSize * outputCols)
        MSE /= (validSize * outputCols)
        accuracy /= validSize

        # print evaluation result
        resultSummary = '----------------\n'
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
# normalizeTarget : normalize target(output) value?

# variables not given as argument of this function
# fn              : [train input file name, train output file name, test input file name, test output file name]
# normalizeName   : normalize info (average and stddev) file name
def dataFromDF(dfTrain, dfTest, targetColName, exceptCols, normalizeTarget):

    # read configuration file
    f = open('config.txt', 'r')
    fl = f.readlines()
    f.close()
    for i in range(len(fl)): fl[i] = fl[i].split('\n')[0]

    # init values as None
    fn = [None, None, None]
    normalizeName = None

    # extract configuration
    # trainInput     : train input data file name
    # trainOutput    : train output data file name
    # testInput      : test input data file name
    for i in range(len(fl)):
        configSplit = fl[i].split(' ') # split

        if configSplit[0] == 'trainInput': fn[0] = configSplit[1]
        elif configSplit[0] == 'trainOutput': fn[1] = configSplit[1]
        elif configSplit[0] == 'testInput': fn[2] = configSplit[1]
        elif configSplit[0] == 'normalizeName': normalizeName = configSplit[1]

        # convert 'None' into None
        if fn[0] == 'None': fn[0] = None
        if fn[1] == 'None': fn[1] = None
        if fn[2] == 'None': fn[2] = None
        if normalizeName == 'None': normalizeName = None

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

    # make train input, train output, and test input array
    dfTrainInputArray = np.array(dfTrainInput)
    dfTrainOutputArray = np.array(dfTrainOutput)
    dfTestInputArray = np.array(dfTestInput)

    # print each dataFrame array
    print('\n *** dataFrame (training input) ***')
    print(dfTrainInputArray)

    print('\n *** dataFrame (test input) ***')
    print(dfTestInputArray)

    print('\n *** dataFrame (training output - not normalized) ***')
    print(dfTrainOutputArray)

    # train output array -> [a, b, c, ...] to [[a], [b], [c], ...]
    dfTrainOutputArray_ = []
    for i in range(len(dfTrainOutputArray)): dfTrainOutputArray_.append([dfTrainOutputArray[i]])

    # make test output file (before normalization)
    saveArray(fn[1].split('.')[0] + '_not_normalized.txt', dfTrainOutputArray_)

    # normalize training output data and write avg and stddev
    if normalizeTarget == True:
        writeNormalizeInfo(dfTrainOutputArray_, normalizeName)

        print('\n *** dataFrame (training output - normalized) ***')
        for i in range(5): print(dfTrainOutputArray_[i])
        print('...')
        for i in range(len(dfTrainOutputArray_)-5, len(dfTrainOutputArray_)): print(dfTrainOutputArray_[i])
        print('')

    # make train input, train output, and test output file
    saveArray(fn[0], dfTrainInputArray)
    saveArray(fn[1], dfTrainOutputArray_)
    saveArray(fn[2], dfTestInputArray)
