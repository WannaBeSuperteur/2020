import sys
sys.path.insert(0, '../../../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# read whole training and test data and write
# train_id, train_input, train_output, test_id and test_input
# train_id_sub_X, train_input_sub_X, train_output_sub_X
# test_id_sub_X, test_input_sub_X
def readAllSubs():
    
    # read train and test data
    train = RD.loadArray('train.csv', ',')
    test = RD.loadArray('test.csv', ',')

    # write id-delta, input and output of training data
    # write id-delta and input         of test     data
    try:
        _ = open('train_id.txt', 'r')
        _.close()
        _ = open('train_input.txt', 'r')
        _.close()
        _ = open('train_output.txt', 'r')
        _.close()
        _ = open('test_id.txt', 'r')
        _.close()
        _ = open('test_input.txt', 'r')
        _.close()
        
    except:
        # train.txt -> id, delta, start1~400, stop1~400 -> train_id.txt     : extract id and delta
        #                                               -> train_input.txt  : extract delta and stop1~400
        #                                               -> train_output.txt : extract delta and start1~400
        RD.saveArray('train_id.txt', np.array(train)[:, 0:2])
        RD.saveArray('train_input.txt', np.concatenate([np.array(train)[:, 1:2], np.array(train)[:, 402:802]], axis=1))
        RD.saveArray('train_output.txt', np.array(train)[:, 1:402])

        # test.txt  -> id, delta, stop1~400             -> test_id.txt      : extract id and delta
        #                                               -> test_input.txt   : extract delta and stop1~400
        RD.saveArray('test_id.txt', np.array(test)[:, 0:2])
        RD.saveArray('test_input.txt', np.array(test)[:, 1:402])

    # split train and test data into files
    try:
        # try to read file
        for i in range(5):
            _ = open('train_id_sub_' + str(i) + '.txt', 'r')
            _.close()
            _ = open('train_input_sub_' + str(i) + '.txt', 'r')
            _.close()
            _ = open('train_output_sub_' + str(i) + '.txt', 'r')
            _.close()
            _ = open('test_id_sub_' + str(i) + '.txt', 'r')
            _.close()
            _ = open('test_input_sub_' + str(i) + '.txt', 'r')
            _.close()
    except:
        # write train_id, train_input, train_output, test_id and test_input files
        deltaOrder = [[1], [2], [3], [4], [5]] # order of delta (1, 2, 3, 4, 5)

        # train_id_sub_X.txt     : id             of training data with delta X
        # train_input_sub_X.txt  : input  (stop)  of training data with delta X
        # train_output_sub_X.txt : output (start) of training data with delta X
        # test_id_sub_X.txt      : id             of test data with delta X
        # test_input_sub_X.txt   : input  (stop)  of test data with delta X
        RD.splitArray('train_id.txt', [1], deltaOrder, True)
        RD.splitArray('train_input.txt', [0], deltaOrder, True)
        RD.splitArray('train_output.txt', [0], deltaOrder, True)
        RD.splitArray('test_id.txt', [1], deltaOrder, True)
        RD.splitArray('test_input.txt', [0], deltaOrder, True)

# make input(n*n) and output data(1*1), called 'n-sub mode'
# delta          : delta value
# n              : size of training data
# size           : size of original board
# limitLen       : trainLen = min(train input file size, limitLen)
# writeTestInput : write test input?
def makeData(delta, n, size, limitLen, writeTestInput):

    # window size
    ws = int((n-1)/2)

    # read data
    trainInput = RD.loadArray('train_input_sub_' + str(delta-1) + '.txt')
    trainOutput = RD.loadArray('train_output_sub_' + str(delta-1) + '.txt')
    testInput = RD.loadArray('test_input_sub_' + str(delta-1) + '.txt')
    
    trainLen = min(len(trainInput), limitLen)
    testLen = len(testInput)

    # input data to make
    trainInputData = []

    # output data to make
    trainOutputData = []

    # test input data to make
    if writeTestInput == True: testInputData = []

    # reshape training data
    for i in range(trainLen):
        if i % 10 == 0: print('makeData (training) : ' + str(i) + ' / ' + str(trainLen))

        # trainInput and trainOutput as numeric type
        trainInput = np.array(trainInput).astype('float')
        trainOutput = np.array(trainOutput).astype('float')

        # reshape to derive n*n training data (with ws-sized padding)
        thisReshaped = np.pad(np.array(trainInput[i]).reshape(size, size), ((ws, ws), (ws, ws)), 'constant', constant_values=-1)
        
        # save training data into array trainInputData and trainOutputData
        for j in range(ws, size+ws):
            for k in range(ws, size+ws):
                trainInputData.append(list(thisReshaped[j-ws:j+ws+1, k-ws:k+ws+1].reshape(n*n)))
                trainOutputData.append([trainOutput[j][k]])

    # reshape test data
    if writeTestInput == True:
        for i in range(testLen):
            if i % 10 == 0: print('makeData (test) : ' + str(i) + ' / ' + str(testLen))

            # trainInput and trainOutput as numeric type
            testInput = np.array(testInput).astype('float')

            # reshape to derive n*n training data (with ws-sized padding)
            thisReshaped = np.pad(np.array(testInput[i]).reshape(size, size), ((ws, ws), (ws, ws)), 'constant', constant_values=-1)
            
            # save test data into array testInputData
            for j in range(ws, size+ws):
                for k in range(ws, size+ws):
                    testInputData.append(list(thisReshaped[j-ws:j+ws+1, k-ws:k+ws+1].reshape(n*n)))

    # save as file
    RD.saveArray('train_input_n_sub_' + str(delta-1) + '.txt', trainInputData)
    RD.saveArray('train_output_n_sub_' + str(delta-1) + '.txt', trainOutputData)
    if writeTestInput == True: RD.saveArray('test_input_n_sub_' + str(delta-1) + '.txt', testInputData)

if __name__ == '__main__':
    np.set_printoptions(edgeitems=1000, linewidth=10000)

    readAllSubs()
    makeData(1, 5, 20, 1000, False)
    makeData(2, 7, 20, 1000, False)
    makeData(3, 9, 20, 1000, False)
    makeData(4, 11, 20, 1000, False)
    makeData(5, 13, 20, 1000, False)
