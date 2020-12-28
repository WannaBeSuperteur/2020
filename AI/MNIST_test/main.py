import helper
import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# function to read image from csv file
# colRange: [a, None] - retrieve columns with indices a, ..., (last column)
#           [a, b]    - retrieve columns with indices a, ..., b-1
# rowRange: [a, None] - retrieve rows    with indices a, ..., (last row)
#           [a, b]    - retrieve rows    with indices a, ..., b-1
def readCSV(fn, colRange, rowRange):
    csv = RD.loadArray(fn, ',')

    # colRange start and end, rowRange start and end
    cs = colRange[0]
    ce = colRange[1]
    rs = rowRange[0]
    re = rowRange[1]

    # colRange: [a, None]
    if ce == None:

        # rowRange: [a, None]
        if re == None: return np.array(csv)[rs:, cs:]

        # rowRange: [a, b]
        else: return np.array(csv)[rs:re, cs:]

    # colRange: [a, b]
    else:

        # rowRange: [a, None]
        if re == None: return np.array(csv)[rs:, cs:ce]

        # rowRange: [a, b]
        else: return np.array(csv)[rs:re, cs:ce]

# convert to numeric value (/255) and save
def convertToNumeric():

    # number of train rows and test rows
    trainRows = 12000 # / 60000
    testRows = 2000 # / 10000

    # check if converted data file exists
    try:
        _ = open('mnist_train_input.txt', 'r')
        _.close()
        _ = open('mnist_train_output.txt', 'r')
        _.close()
        _ = open('mnist_test_input.txt', 'r')
        _.close()
        _ = open('mnist_test_output.txt', 'r')
        _.close()
        return
    except:
        pass
    
    # TRAINING DATA
    # read and print col=1, ... and row=1, ... of the CSV file
    train_input = readCSV('mnist_train.csv', [1, None], [1, trainRows+1])
    train_output = readCSV('mnist_train.csv', [0, 1], [1, trainRows+1])

    # TEST DATA
    test_input = readCSV('mnist_test.csv', [1, None], [1, testRows+1])
    test_output = readCSV('mnist_test.csv', [0, 1], [1, testRows+1])

    # make training data numeric
    for i in range(trainRows):
        if i % 1000 == 0: print(i)
        
        for j in range(len(train_input[0])):
            train_input[i][j] = int(train_input[i][j]) / 255

    # make test data numeric
    for i in range(testRows):
        if i % 1000 == 0: print(i)
        
        for j in range(len(test_input[0])):
            test_input[i][j] = int(test_input[i][j]) / 255

    # save
    RD.saveArray('mnist_train_input.txt', train_input)
    RD.saveArray('mnist_train_output.txt', train_output)
    RD.saveArray('mnist_test_input.txt', test_input)
    RD.saveArray('mnist_test_output.txt', test_output)

# read training and test data
def readMNISTData():
    print('reading MNIST data...')
    
    # TRAINING DATA
    # read and print col=1, ... and row=1, ... of the CSV file
    train_input = np.array(RD.loadArray('mnist_train_input.txt'))
    train_output = np.array(RD.loadArray('mnist_train_output.txt'))
    print(train_input)
    print(train_output)

    # TEST DATA
    test_input = np.array(RD.loadArray('mnist_test_input.txt'))
    test_output = np.array(RD.loadArray('mnist_test_output.txt'))
    print(test_input)
    print(test_output)

    return (train_input, train_output, test_input, test_output)

# main
if __name__ == '__main__':

    # meta info
    TRI = 'mnist_train_input.txt'
    TRO = 'mnist_train_output.txt'
    TEI = 'mnist_test_input.txt'
    TEO = 'mnist_test_predict.txt'

    TE_real = 'mnist_test_output.txt'
    TE_report = 'mnist_test_report.txt'
    VAL_rate = 0.0
    VAL_report = 'mnist_val_report.txt'
    modelConfig = 'mnist_model_config.txt'

    # preprocess data
    convertToNumeric()

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # training and test
    # 'config.txt' is used for configuration
    DL.deepLearning(TRI, TRO, TEI, TEO,
                    TE_real, TE_report, VAL_rate, VAL_report, modelConfig,
                    deviceName, epoch, printed, 'model')
