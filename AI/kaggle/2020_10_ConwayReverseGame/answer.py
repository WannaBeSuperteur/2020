import sys
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD
import deepLearning_main as DL

if __name__ == '__main__':
    np.set_printoptions(edgeitems=30, linewidth=180)
    verbose = False

    # ~_sub_0.txt : with delta 1
    # ~_sub_1.txt : with delta 2
    # ~_sub_2.txt : with delta 3
    # ~_sub_3.txt : with delta 4
    # ~_sub_4.txt : with delta 5

    # save real training and test data to use deep learning, for each case delta=1,2,3,4 and 5
    for i in range(5):

        # file names and configurations
        trainIName = 'train_input_sub_' + str(i) + '.txt'
        trainOName = 'train_output_sub_' + str(i) + '.txt'
        testIName = 'test_input_sub_' + str(i) + '.txt'
        testOName = 'test_output_sub_' + str(i) + '.txt' # file to make
        testReport = 'test_report_sub_' + str(i) + '.txt' # file to make
        validRate = 0.05
        validReport = 'valid_report_sub_' + str(i) + '.txt' # file to make
        modelConfig = 'model_sub_' + str(i) + '.txt' # file to make

        deviceName = 'cpu:0'
        epoch = 20

        # load arrays
        train_id_list = RD.loadArray('train_id_sub_' + str(i) + '.txt')
        trainI_array = RD.loadArray(trainIName)
        trainO_array = RD.loadArray(trainOName)

        test_id_list = RD.loadArray('test_id_sub_' + str(i) + '.txt')
        testI_array = RD.loadArray(testIName)

        # print training and test array
        if verbose == True:
            for j in range(5):
                print('\ntrainI_array = stop - delta=' + str(i+1) + ', index=' + str(j))
                print(np.array(trainI_array)[j].reshape(20, 20))
                print('\ntrainO_array = start - delta=' + str(i+1) + ', index=' + str(j))
                print(np.array(trainO_array)[j].reshape(20, 20))
                print('\ntestI_array = stop - delta=' + str(i+1) + ', index=' + str(j))
                print(np.array(testI_array)[j].reshape(20, 20))

        # deep learning : test array -> training array
        DL.deepLearning(trainIName, trainOName, testIName, testOName, None, testReport,
                        validRate, validReport, modelConfig, deviceName, epoch, verbose, 'model_sub_' + str(i))
