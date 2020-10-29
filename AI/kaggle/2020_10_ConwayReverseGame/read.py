import sys
sys.path.insert(0, '../../../../AI_BASE')
import readData as RD

import numpy as np

if __name__ == '__main__':

    # read train and test data
    try:
        _ = open('train.txt', 'r')
        _ = open('test.txt', 'r')
    except:
        train = RD.loadArray('train.csv', ',')
        test = RD.loadArray('test.csv', ',')

        print('train')
        print(np.array(train))
        print('test')
        print(np.array(test))

        RD.saveArray('train.txt', train)
        RD.saveArray('test.txt', test)

    # split train and test data into 5 files
    split_train = RD.splitArray('train.txt', [1], True)
    split_test = RD.splitArray('test.txt', [1], True)

    print('split_train=' + str(split_train))
    print('split_test=' + str(split_test))
