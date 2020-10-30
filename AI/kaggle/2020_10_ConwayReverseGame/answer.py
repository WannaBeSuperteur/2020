import sys
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD
import deepLearning_main as DL

if __name__ == '__main__':

    # delta = 1, 2, 3, 4 and 5
    trainNames = ['train_sub_4.txt', 'train_sub_2.txt', 'train_sub_3.txt', 'train_sub_0.txt', 'train_sub_1.txt']
    testNames = ['test_sub_2.txt', 'test_sub_0.txt', 'test_sub_4.txt', 'test_sub_3.txt', 'test_sub_1.txt']

    for i in range(5):

        # configuration
        trainName = trainNames[i]
        testName = testNames[i]
        ftype = 'txt'

        fcolsTrain = ['id']
        fcolsTest = ['id']
        for j in range(400): fcolsTrain.append('start' + str(j))
        for j in range(400): fcolsTrain.append('stop' + str(j))
        for j in range(400): fcolsTest.append('stop' + str(j))

        # execute deep learning
        DL.deepLearning(trainName, testName, 
