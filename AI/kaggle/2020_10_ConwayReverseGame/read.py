import sys
sys.path.insert(0, '../../../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# file input and output
# input  : train.csv, test.csv
# output : train_id.txt, train_input.txt, train_output.txt, test_id.txt, test_input.txt,
#          train_id_sub_X.txt, train_input_sub_X.txt, train_output_sub_X.txt, test_id_sub_X.txt, test_input_sub_X.txt

# files read by this code
# train.csv              (loadArray in readAllSubs)
# test.csv               (loadArray in readAllSubs)
# train_id.txt           (open in readAllSubs)
# train_input.txt        (open in readAllSubs)
# train_output.txt       (open in readAllSubs)
# test_id.txt            (open in readAllSubs)
# test_input.txt         (open in readAllSubs)
# train_id_sub_X.txt     (open in readAllSubs)
# train_input_sub_X.txt  (open in readAllSubs)
# train_output_sub_X.txt (open in readAllSubs)
# test_id_sub_X.txt      (open in readAllSubs)
# test_input_sub_X.txt   (open in readAllSubs)
# train_input_sub_X.txt  (loadArray in makeData)
# train_output_sub_X.txt (loadArray in makeData)
# test_input_sub_X.txt   (loadArray in makeData)

# files written by this code
# [except]            train_id.txt             (saveArray in readAllSubs)
# [except]            train_input.txt          (saveArray in readAllSubs)
# [except]            train_output.txt         (saveArray in readAllSubs)
# [except]            test_id.txt              (saveArray in readAllSubs)
# [except]            test_input.txt           (saveArray in readAllSubs)
# [except]            train_id_sub_X.txt       (splitArray in readAllSubs)
# [except]            train_input_sub_X.txt    (splitArray in readAllSubs)
# [except]            train_output_sub_X.txt   (splitArray in readAllSubs)
# [except]            test_id_sub_X.txt        (splitArray in readAllSubs)
# [except]            test_input_sub_X.txt     (splitArray in readAllSubs)
#                     train_input_n_sub_X.txt  (saveArray in makeData)
#                     train_output_n_sub_X.txt (saveArray in makeData)
#          [optional] test_input_n_sub_X.txt   (saveArray in makeData)

# not 'n-sub mode'
# read whole training and test data and write
# train_id, train_input, train_output, test_id and test_input
# train_id_sub_X, train_input_sub_X, train_output_sub_X
# test_id_sub_X, test_input_sub_X
def readAllSubs(size):
    
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
        # train.txt -> id, delta, start1~400, stop1~400 (if size=20) -> train_id.txt     : extract id and delta
        #                                                            -> train_input.txt  : extract delta and stop1~400 (if size=20)
        #                                                            -> train_output.txt : extract delta and start1~400 (if size=20)
        RD.saveArray('train_id.txt', np.array(train)[:, 0:2])
        RD.saveArray('train_input.txt', np.concatenate([np.array(train)[:, 1:2], np.array(train)[:, size*size+2:2*size*size+2]], axis=1))
        RD.saveArray('train_output.txt', np.array(train)[:, 1:size*size+2])

        # test.txt  -> id, delta, stop1~400 (if size=20)             -> test_id.txt      : extract id and delta
        #                                                            -> test_input.txt   : extract delta and stop1~400 (if size=20)
        RD.saveArray('test_id.txt', np.array(test)[:, 0:2])
        RD.saveArray('test_input.txt', np.array(test)[:, 1:size*size+2])

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
# n              : size of training/test input data (width or height)
# n_             : size of training/test output data (width or height)
# size           : size of original board
# limitLen       : trainLen = min(train input file size, limitLen)
# writeTestInput : write test input?
def makeData(delta, n, n_, size, limitLen, writeTestInput):

    # window size
    ws = int((n-1)/2) # for training/test input
    ws_ = int((n_-1)/2) # for training/test output

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
        trainInputReshaped = np.pad(np.array(trainInput[i]).reshape(size, size), ((ws, ws), (ws, ws)), 'constant', constant_values=-1)
        trainOutputReshaped = np.pad(np.array(trainOutput[i]).reshape(size, size), ((ws_, ws_), (ws_, ws_)), 'constant', constant_values=-1)
        
        # save training data into array trainInputData and trainOutputData
        for j in range(size):
            for k in range(size):
                trainInputData.append(list(trainInputReshaped[j:j+2*ws+1, k:k+2*ws+1].reshape(n*n)))
                trainOutputData.append(list(trainOutputReshaped[j:j+2*ws_+1, k:k+2*ws_+1].reshape(n_*n_)))

    # reshape test data
    if writeTestInput == True:
        for i in range(testLen):
            if i % 10 == 0: print('makeData (test) : ' + str(i) + ' / ' + str(testLen))

            # trainInput and trainOutput as numeric type
            testInput = np.array(testInput).astype('float')

            # reshape to derive n*n training data (with ws-sized padding)
            testInputReshaped = np.pad(np.array(testInput[i]).reshape(size, size), ((ws, ws), (ws, ws)), 'constant', constant_values=-1)
            
            # save test data into array testInputData
            for j in range(size):
                for k in range(size):
                    testInputData.append(list(testInputReshaped[j:j+2*ws+1, k:k+2*ws+1].reshape(n*n)))

    # save as file
    RD.saveArray('train_input_n_sub_' + str(delta-1) + '.txt', trainInputData)
    RD.saveArray('train_output_n_sub_' + str(delta-1) + '.txt', trainOutputData)
    if writeTestInput == True: RD.saveArray('test_input_n_sub_' + str(delta-1) + '.txt', testInputData)

if __name__ == '__main__':
    np.set_printoptions(edgeitems=1000, linewidth=10000)

    size = 20 # the number of rows/columns in each input data
    outputSize = 1 # the number of rows/columns in each output data

    # for normal (not n-sub) mode
    readAllSubs(size)

    # for n-sub mode
    # it takes 2~3 hours in total
    makeData(1, 5, outputSize, size, 1000, False)
    makeData(2, 7, outputSize, size, 1000, False)
    makeData(3, 9, outputSize, size, 1000, False)
    makeData(4, 11, outputSize, size, 1000, False)
    makeData(5, 13, outputSize, size, 1000, False)
