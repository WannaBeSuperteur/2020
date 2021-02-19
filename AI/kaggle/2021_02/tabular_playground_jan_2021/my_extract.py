import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    train_rows = 300000
    test_rows = 200000
    input_cols = 14

    result = ''

    # TRAIN
    train_data = RD.loadArray('train.csv', ',')
    train_input = np.array(train_data)[1:, 1:15]
    train_output = np.array(train_data)[1:, 15:]

    # AVG and STD for each column of training input
    avgs = []
    stds = []

    for i in range(input_cols):
        thisCol = train_input[:, i].astype(float)

        # compute avg and stddev
        avgs.append(np.mean(thisCol))
        stds.append(np.std(thisCol))

    result += 'average:\n'
    result += str(avgs) + '\n'
    result += 'stddevs:\n'
    result += str(stds) + '\n'

    # convert training input data using average and stddev
    # convert training output data: x' = x - 8
    for i in range(train_rows):
        for j in range(input_cols):
            train_input[i][j] = (float(train_input[i][j]) - avgs[j]) / stds[j]
        train_output[i][0] = float(train_output[i][0]) - 8

    result += 'train input:\n'
    result += str(np.array(train_input)) + '\n'
    result += 'train output:\n'
    result += str(np.array(train_output)) + '\n'

    # save training data
    RD.saveArray('train_input.txt', train_input, '\t', 500)
    RD.saveArray('train_output.txt', train_output, '\t', 500)

    # TEST
    test_data = RD.loadArray('test.csv', ',')
    test_input = np.array(test_data)[1:, 1:]
    
    result += 'test input:\n'
    result += str(np.array(test_input)) + '\n'

    # convert test input data using average and stddev of training input
    for i in range(test_rows):
        for j in range(input_cols):
            test_input[i][j] = (float(test_input[i][j]) - avgs[j]) / stds[j]

    # save test data
    RD.saveArray('test_input.txt', test_input, '\t', 500)

    # save the result
    f = open('my_extract_result.txt', 'w')
    f.write(result)
    f.close()
