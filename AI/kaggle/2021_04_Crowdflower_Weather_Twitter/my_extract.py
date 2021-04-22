import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import pandas as pd

# write input data
def writeInput(array_input, state_list):

    final_input = []

    for i in range(len(array_input)):
        thisRow = []

        # tweet (remove all the tabs)
        thisRow.append(array_input[i][0].replace('\t', ''))

        # state
        for j in range(len(state_list)):
            if array_input[i][1] == state_list[j]:
                thisRow.append(1)
            else:
                thisRow.append(falseVal)

        final_input.append(thisRow)

    return final_input

if __name__ == '__main__':

    # configuration
    falseVal = -1

    # create train input and output
    train_data = np.array(pd.read_csv('train.csv'))
    train_input = train_data[1:, 1:4]
    train_output = train_data[1:, 4:]

    test_data = np.array(pd.read_csv('test.csv'))
    test_input = test_data[1:, 1:4]

    # one-hot for state
    state_list = list(set(train_data[:, 2]))
    print(state_list)

    # write final train_input and test_input
    # test output is as original
    final_train_input = writeInput(train_input, state_list)
    final_test_input = writeInput(test_input, state_list)

    # save training and test data
    RD.saveArray('train_input.txt', final_train_input, '\t', 500)
    RD.saveArray('train_output.txt', train_output, '\t', 500)
    RD.saveArray('test_input.txt', final_test_input, '\t', 500)

    # split training input into train_train and train_valid
    validRate = 0.15
    trainRows = len(final_train_input)

    train_train_input = []
    train_train_output = []
    train_valid_input = []
    train_valid_output = []

    for i in range(trainRows):
        if i >= validRate * trainRows: # train_train
            train_train_input.append(final_train_input[i])
            train_train_output.append(train_output[i])
            
        else: # train_valid
            train_valid_input.append(final_train_input[i])
            train_valid_output.append(train_output[i])

    # save splitted training data
    RD.saveArray('train_train_input.txt', train_train_input, '\t', 500)
    RD.saveArray('train_train_output.txt', train_train_output, '\t', 500)
    RD.saveArray('train_valid_input.txt', train_valid_input, '\t', 500)
    RD.saveArray('train_valid_output.txt', train_valid_output, '\t', 500)
