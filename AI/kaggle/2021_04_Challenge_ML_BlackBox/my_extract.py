import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    train_rows = 1000
    test_rows = 10000
    input_cols = 1875

    result = ''

    # TRAIN
    train_data = RD.loadArray('train.csv', ',')
    train_input = np.array(train_data)[1:, 1:1876]
    train_output_raw = np.array(train_data)[1:, 0:1]
    train_output = []

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
            new_input = (float(train_input[i][j]) - avgs[j]) / stds[j]
            if abs(new_input) < 0.001: new_input = 0
            train_input[i][j] = new_input

        # one-hot: 1 -> [1, 0, ..., 0, 0]
        #          2 -> [0, 1, ..., 0, 0]
        #          ...
        #          9 -> [0, 0, ..., 0, 1]
        train_output_temp = []
        
        for j in range(9):
            if int(float(train_output_raw[i][0])) == j + 1: train_output_temp.append(1)
            else: train_output_temp.append(0)
            
        train_output.append(train_output_temp)

    # convert train output to numpy array
    train_output = np.array(train_output)

    result += 'train input:\n'
    result += str(np.array(train_input)) + '\n'
    result += 'train output:\n'
    result += str(np.array(train_output)) + '\n'

    # save training data
    RD.saveArray('train_input.txt', train_input, '\t', 50)
    RD.saveArray('train_output.txt', train_output, '\t', 50)

    # TEST
    test_data = RD.loadArray('test.csv', ',')
    test_input = np.array(test_data)[1:, :]
    
    result += 'test input:\n'
    result += str(np.array(test_input)) + '\n'

    # convert test input data using average and stddev of training input
    for i in range(test_rows):
        for j in range(input_cols):
            new_input = (float(test_input[i][j]) - avgs[j]) / stds[j]
            if abs(new_input) < 0.001: new_input = 0
            test_input[i][j] = new_input

    # save test data
    RD.saveArray('test_input.txt', test_input, '\t', 50)

    # save the result
    f = open('my_extract_result.txt', 'w')
    f.write(result)
    f.close()
