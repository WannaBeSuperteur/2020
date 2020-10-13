import deepLearning_main as DL
import deepLearning_GPU_helper as helper
import numpy as np
import random
    
def AIbase_deeplearning():

    # read configuration file
    f = open('config.txt', 'r')
    fl = f.readlines()
    f.close()
    for i in range(len(fl)): fl[i] = fl[i].split('\n')[0]

    # init values as None
    trainInputFile = None
    trainOutputFile = None
    testInputFile = None
    testOutputFile = None
    testOutputReal = None
    test_report = None
    valid_rate = None
    valid_report = None
    modelConfig = None
    normalizeName = None

    # extract configuration
    # trainInput     : train input data file name
    # trainOutput    : train output data file name
    # testInput      : test input data file name
    # testOutput     : test output (prediction) data file name
    # testOutputReal : test output (real) data file name (if not None, compare output prediction with real test output data)
    # test_report    : test report file name
    # valid_rate     : rate of validation data among training data (0 -> use test data)
    # valid_report   : vaildation report file name
    # modelConfig    : model configuration file name
    # normalizeName  : normalize info (average and stddev) file name
    for i in range(len(fl)):
        configSplit = fl[i].split(' ') # split

        if configSplit[0] == 'trainInput': trainInputFile = configSplit[1]
        elif configSplit[0] == 'trainOutput': trainOutputFile = configSplit[1]
        elif configSplit[0] == 'testInput': testInputFile = configSplit[1]
        elif configSplit[0] == 'testOutput': testOutputFile = configSplit[1]
        elif configSplit[0] == 'testOutputReal': testOutputReal = configSplit[1]
        elif configSplit[0] == 'test_report': test_report = configSplit[1]
        elif configSplit[0] == 'valid_rate': valid_rate = float(configSplit[1])
        elif configSplit[0] == 'valid_report': valid_report = configSplit[1]
        elif configSplit[0] == 'modelConfig': modelConfig = configSplit[1]
        elif configSplit[0] == 'normalizeName': normalizeName = configSplit[1]

        # convert 'None' to None
        if trainInputFile == 'None': trainInputFile = None
        if trainOutputFile == 'None': trainOutputFile = None
        if testInputFile == 'None': testInputFile = None
        if testOutputFile == 'None': testOutputFile = None
        if testOutputReal == 'None': testOutputReal = None
        if test_report == 'None': test_report = None
        if valid_rate == 'None': valid_rate = None
        if valid_report == 'None': valid_report = None
        if modelConfig == 'None': modelConfig = None
        if normalizeName == 'None': normalizeName = None

    # do Deep Learning
    # if the file for training/test data already exists, just write prediction for deep learning
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    print('\n ### deep learning ###')
    DL.deepLearning(trainInputFile, trainOutputFile, testInputFile, testOutputFile, testOutputReal,
                    test_report, valid_rate, valid_report, modelConfig, deviceName, epoch, printed, 'model')

if __name__ == '__main__':
    AIbase_deeplearning()
