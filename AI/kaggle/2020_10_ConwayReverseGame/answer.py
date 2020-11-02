import sys
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD
import deepLearning_main as DL
import MAE_for_CRGoL as MAE

if __name__ == '__main__':
    np.set_printoptions(edgeitems=30, linewidth=180)
    verbose = False
    use_n_sub = True # using 'train_input/output_n_sub_X.txt' as training input/output data

    # ~_sub_0.txt : with delta 1
    # ~_sub_1.txt : with delta 2
    # ~_sub_2.txt : with delta 3
    # ~_sub_3.txt : with delta 4
    # ~_sub_4.txt : with delta 5

    # threshold list to find MAE
    thresholdList = []
    for i in range(1, 100): thresholdList.append(round(0.01*i, 6))

    # save real training and test data to use deep learning, for each case delta=1,2,3,4 and 5
    for i in range(5):

        validRate = 0.05
        deviceName = 'cpu:0'
        epoch = 20

        # file names and configurations
        if use_n_sub = True: # use n-sub mode

        else: # do not use n-sub mode
            trainIName = 'train_input_sub_' + str(i) + '.txt'
            trainOName = 'train_output_sub_' + str(i) + '.txt'
            testIName = 'test_input_sub_' + str(i) + '.txt'
            testOName = 'test_output_sub_' + str(i) + '.txt' # file to make
            testReport = 'test_report_sub_' + str(i) + '.txt' # file to make
            validReport = 'valid_report_sub_' + str(i) + '.txt' # file to make
            modelConfig = 'model_sub_' + str(i) + '.txt' # file to make
            modelName = 'model_sub_' + str(i) # model name

        # load arrays
        if use_n_sub == True: # use n-sub mode

        else: # do not use n-sub mode
            train_id_list = RD.loadArray('train_id_sub_' + str(i) + '.txt')
            trainI_array = RD.loadArray(trainIName)
            trainO_array = RD.loadArray(trainOName)

            test_id_list = RD.loadArray('test_id_sub_' + str(i) + '.txt')
            testI_array = RD.loadArray(testIName)

        # print training and test array
        if verbose == True:
            for j in range(5):
                trainI_ = np.array(trainI_array)[j]
                trainO_ = np.array(trainO_array)[j]
                testI_ = np.array(testI_array)[j]
                
                print('\ntrainI_array = stop - delta=' + str(i+1) + ', index=' + str(j))
                print(trainI_.reshape(int(math.sqrt(len(trainI_))), int(math.sqrt(len(trainI_)))))
                print('\ntrainO_array = start - delta=' + str(i+1) + ', index=' + str(j))
                print(trainO_.reshape(int(math.sqrt(len(trainO_))), int(math.sqrt(len(trainO_)))))
                print('\ntestI_array = stop - delta=' + str(i+1) + ', index=' + str(j))
                print(testI_.reshape(int(math.sqrt(len(testI_))), int(math.sqrt(len(testI_)))))

        # deep learning : test array -> training array
        DL.deepLearning(trainIName, trainOName, testIName, testOName, None, testReport,
                        validRate, validReport, modelConfig, deviceName, epoch, verbose, modelName)

        # print MAE
        MAE.readValidReport(validReport, thresholdList, 400)
