import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':
    train_tokenized = np.array(RD.loadArray('train_tokenized.txt')).astype(float)
    valid_tokenized = np.array(RD.loadArray('valid_tokenized.txt')).astype(float)

    print('train_tokenized :')
    print(train_tokenized)
    print('\ntest_tokenized :')
    print(valid_tokenized)

    # number of columns
    cols = len(train_tokenized[0])

    max_length_train = 0
    max_length_test = 0
    max_length = 0

    # find the number of nonzero columns (=max_length)
    while max(train_tokenized[:, max_length_train]) > 0:
        max_length_train += 1

    while max(valid_tokenized[:, max_length_test]) > 0:
        max_length_test += 1

    max_length = max(max_length_train, max_length_test)
        
    # print the final answer
    print('max_length for train = ' + str(max_length_train))
    print('max_length for test = ' + str(max_length_test))
    print('max_length = ' + str(max_length))
