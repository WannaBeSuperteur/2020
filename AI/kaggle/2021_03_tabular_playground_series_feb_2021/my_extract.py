import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    train_rows = 300000
    test_rows = 200000
    input_cols_cat = 10
    input_cols_cont = 14

    result = ''

    # TRAIN
    train_data = RD.loadArray('train.csv', ',')
    train_input = np.array(train_data)[1:, 1:25]
    train_output = np.array(train_data)[1:, 25:]

    # AVG and STD for each column of CONTINUOUS training input
    cont_avgs = []
    cont_stds = []

    # categorical
    for i in range(input_cols_cat):
        cont_avgs.append(-1)
        cont_stds.append(-1)

    # continuous
    for i in range(input_cols_cat, input_cols_cat + input_cols_cont):
        thisCol = train_input[:, i].astype(float)

        # compute avg and stddev
        cont_avgs.append(np.mean(thisCol))
        cont_stds.append(np.std(thisCol))

    result += 'average:\n'
    result += str(cont_avgs) + '\n'
    result += 'stddevs:\n'
    result += str(cont_stds) + '\n'

    # make the set of values for categorial variables
    catg_set = []
    for i in range(input_cols_cat):
        catg_set.append(list(set(train_input[:, i])))

    # make final train input and final train output
    final_train_input = []
    final_train_output = []

    for i in range(train_rows):
        if i % 1000 == 0: print(i)

        train_thisRow = []

        # categorial -> one-hot with 0 as -1
        # ['A', 'B', 'C', 'A'] -> [[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]]
        for j in range(input_cols_cat):
            for k in range(len(catg_set[j])):
                if catg_set[j][k] == train_input[i][j]:
                    train_thisRow.append(1)
                else:
                    train_thisRow.append(-1)                

        # continuous -> use avg and std
        # convert training input data using average and stddev
        for j in range(input_cols_cat, input_cols_cat + input_cols_cont):
            train_thisRow.append((float(train_input[i][j]) - cont_avgs[j]) / cont_stds[j])

        final_train_input.append(train_thisRow)

        # convert training output data: x' = x - 7.5
        final_train_output.append([float(train_output[i][0]) - 7.5])

    result += 'train input:\n'
    result += str(np.array(final_train_input)) + '\n'
    result += 'train output:\n'
    result += str(np.array(final_train_output)) + '\n'

    # save training data
    RD.saveArray('train_input.txt', final_train_input, '\t', 500)
    RD.saveArray('train_output.txt', final_train_output, '\t', 500)

    # TEST
    test_data = RD.loadArray('test.csv', ',')
    test_input = np.array(test_data)[1:, 1:]
    
    result += 'test input:\n'
    result += str(np.array(test_input)) + '\n'

    # make final test input
    final_test_input = []

    for i in range(test_rows):
        if i % 1000 == 0: print(i)

        test_thisRow = []

        # categorial -> one-hot with 0 as -1
        # ['A', 'B', 'C', 'A'] -> [[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]]
        for j in range(input_cols_cat):
            for k in range(len(catg_set[j])):
                if catg_set[j][k] == test_input[i][j]:
                    test_thisRow.append(1)
                else:
                    test_thisRow.append(-1)                

        # continuous -> use avg and std
        # convert test input data using average and stddev of TRAINING DATA
        for j in range(input_cols_cat, input_cols_cat + input_cols_cont):
            test_thisRow.append((float(test_input[i][j]) - cont_avgs[j]) / cont_stds[j])

        final_test_input.append(test_thisRow)

    # save test data
    RD.saveArray('test_input.txt', final_test_input, '\t', 500)

    # save the result
    f = open('my_extract_result.txt', 'w')
    f.write(result)
    f.close()
