import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    train_rows = 2400
    test_rows = 600
    input_cols = 11

    result = ''

    # TRAIN
    train_data = RD.loadArray('train.csv', ',')
    train_input = np.array(train_data)[1:, 1:12]
    train_output0 = np.array(train_data)[1:, 12:13] # formation_energy_ev_natom
    train_output1 = np.array(train_data)[1:, 13:14] # bandgap_energy_ev

    # AVG and STD for each column of CONTINUOUS training input
    cont_avgs = []
    cont_stds = []

    # continuous
    for i in range(input_cols):
        thisCol = train_input[:, i].astype(float)

        # compute avg and stddev
        cont_avgs.append(np.mean(thisCol))
        cont_stds.append(np.std(thisCol))

    result += 'average:\n'
    result += str(cont_avgs) + '\n'
    result += 'stddevs:\n'
    result += str(cont_stds) + '\n'

    # make final train input and final train output
    final_train_input = []
    final_train_output0 = []
    final_train_output1 = []

    for i in range(train_rows):
        if i % 1000 == 0: print(i)

        train_thisRow = []
        
        # convert training input data using average and stddev
        for j in range(input_cols):
            train_thisRow.append((float(train_input[i][j]) - cont_avgs[j]) / cont_stds[j])

        final_train_input.append(train_thisRow)

        # convert using AVG and STD of output data

        # formation_energy_ev_natom
        # convert training output data (0): x0' = (x0 - 0.187614) / 0.104078
        
        # bandgap_energy_ev
        # convert training output data (1): x1' = (x1 - 2.077205) / 1.006635
        final_train_output0.append([(float(train_output0[i][0]) - 0.187614) / 0.104078])
        final_train_output1.append([(float(train_output1[i][0]) - 2.077205) / 1.006635])

    result += 'train input:\n'
    result += str(np.array(final_train_input)) + '\n'
    result += 'train output 0:\n'
    result += str(np.array(final_train_output0)) + '\n'
    result += 'train output 1:\n'
    result += str(np.array(final_train_output1)) + '\n'

    # save training data
    RD.saveArray('train_input.txt', final_train_input, '\t', 500)
    RD.saveArray('train_output_0.txt', final_train_output0, '\t', 500)
    RD.saveArray('train_output_1.txt', final_train_output1, '\t', 500)

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

        # continuous -> use avg and std
        # convert test input data using average and stddev of TRAINING DATA
        for j in range(input_cols):
            test_thisRow.append((float(test_input[i][j]) - cont_avgs[j]) / cont_stds[j])

        final_test_input.append(test_thisRow)

    # save test data
    RD.saveArray('test_input.txt', final_test_input, '\t', 500)

    # save the result
    f = open('my_extract_result.txt', 'w')
    f.write(result)
    f.close()
