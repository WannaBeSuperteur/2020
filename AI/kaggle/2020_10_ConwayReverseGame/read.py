import sys
sys.path.insert(0, '../../../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

if __name__ == '__main__':
    np.set_printoptions(edgeitems=30, linewidth=180)

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

    # save real training and test data to use deep learning, for each case delta=1,2,3,4 and 5
    for i in range(5):
        train_id_list = RD.loadArray('train_id_sub_' + str(i) + '.txt')
        trainI_array = RD.loadArray('train_input_sub_' + str(i) + '.txt')
        trainO_array = RD.loadArray('train_output_sub_' + str(i) + '.txt')

        test_id_list = RD.loadArray('test_id_sub_' + str(i) + '.txt')
        testI_array = RD.loadArray('test_input_sub_' + str(i) + '.txt')

        # print training and test array
        for j in range(5):
            print('\ntrainI_array = stop')
            print(np.array(trainI_array).reshape(20, 20))
            print('\ntrainO_array = start')
            print(np.array(trainO_array).reshape(20, 20))
            print('\ntestI_array = stop')
            print(np.array(testI_array).reshape(20, 20))

        # deep learning : test array -> training array
        # DL.deepLearning(trainNames[i], 'output_delta' + str(i), testNames[i], 
