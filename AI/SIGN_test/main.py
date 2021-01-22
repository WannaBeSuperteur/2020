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
def readCSV(fn, colRange, rowRange, delimiter=','):
    csv = RD.loadArray(fn, delimiter)

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

# one-hot
def one_hot(array):
    result = []
    
    for i in range(len(array)):
        result.append([0, 0, 0, 0, 0])
        result[i][int(array[i][0])] = 1

    return result

# convert to numeric value (/255) and save
def convertToNumeric():

    # number of train rows and test rows
    trainRows = 62133 # about 85% of whole data
    testRows = 11006 # about 15% of whole data

    # check if converted data file exists
    try:
        _ = open('sign_train_input.txt', 'r')
        _.close()
        _ = open('sign_train_output.txt', 'r')
        _.close()
        _ = open('sign_test_input.txt', 'r')
        _.close()
        _ = open('sign_test_output.txt', 'r')
        _.close()
        return
    except:
        pass
    
    # TRAINING DATA
    # read and print col=1, ... and row=1, ... of the CSV file
    print('reading training input...')
    train_input = readCSV('train.csv', [1, None], [1, trainRows+1])
    
    print('reading training output...')
    train_output = readCSV('train.csv', [0, 1], [1, trainRows+1])

    # TEST DATA
    print('reading test input...')
    test_input = readCSV('test.csv', [1, None], [1, testRows+1])
    
    print('reading test output...')
    test_output = readCSV('test.csv', [0, 1], [1, testRows+1])

    # make training data numeric (inverting the color)
    for i in range(trainRows):
        if i % 1000 == 0: print(i)
        
        for j in range(len(train_input[0])):
            train_input[i][j] = 1.0 - int(train_input[i][j]) / 255.0

    # make test data numeric (inverting the color)
    for i in range(testRows):
        if i % 1000 == 0: print(i)
        
        for j in range(len(test_input[0])):
            test_input[i][j] = 1.0 - int(test_input[i][j]) / 255.0

    # make output one-hot
    train_output = list(train_output)
    test_output = list(test_output)

    train_output = one_hot(train_output)
    test_output = one_hot(test_output)

    # save
    print('saving training input...')
    RD.saveArray('sign_train_input.txt', train_input, splitter='\t', saveSize=1000)
    
    print('saving training output...')
    RD.saveArray('sign_train_output.txt', train_output, splitter='\t', saveSize=1000)
    
    print('saving test input...')
    RD.saveArray('sign_test_input.txt', test_input, splitter='\t', saveSize=1000)
    
    print('saving test output...')
    RD.saveArray('sign_test_output.txt', test_output, splitter='\t', saveSize=1000)

# read training and test data
def readSignData():
    print('reading sign data...')
    
    # TRAINING DATA
    # read and print col=1, ... and row=1, ... of the CSV file
    train_input = np.array(RD.loadArray('sign_train_input.txt'))
    train_output = np.array(RD.loadArray('sign_train_output.txt'))
    print(train_input)
    print(train_output)

    # TEST DATA
    test_input = np.array(RD.loadArray('sign_test_input.txt'))
    test_output = np.array(RD.loadArray('sign_test_output.txt'))
    print(test_input)
    print(test_output)

    return (train_input, train_output, test_input, test_output)

# main
if __name__ == '__main__':

    # meta info
    TRI = 'sign_train_input.txt'
    TRO = 'sign_train_output.txt'
    TEI = 'sign_test_input.txt'
    TEO = 'sign_test_predict.txt'

    TE_real = 'sign_test_output.txt'
    TE_report = 'sign_test_report.txt'
    VAL_rate = 0.0
    VAL_report = 'sign_val_report.txt'
    modelConfig = 'sign_model_config.txt'

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
