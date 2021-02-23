import math
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_main as DL
import readData as RD
import random

if __name__ == '__main__':

    # read data
    regular_array = np.array(pd.read_csv('RegularSeasonCompactResults.csv'))
    tourney_array = np.array(pd.read_csv('TourneyCompactResults.csv'))

    all_array = np.concatenate((regular_array, tourney_array), axis=0)

    # extract raw_result : [Wteam, Lteam, 1], [Lteam, Wteam, 0]
    raw_result = []

    # for [Wteam, Lteam, 1]
    for i in range(len(all_array)):
        raw_result.append([all_array[i][2], all_array[i][4], 1])

    # for [Lteam, Wteam, 0]
    for i in range(len(all_array)):
        raw_result.append([all_array[i][4], all_array[i][2], 0])

    print(raw_result[:50])

    RD.saveArray('raw_result.txt', raw_result, '\t', 500)

    # number of teams
    N = 364
    N_start = 1101

    try:
        _ = open('raw_result_input.txt', 'r')
        _.close()
        _ = open('raw_result_output.txt', 'r')
        _.close()
        
    except:
        raw_result_input = []
        raw_result_output = []

        for i in range(len(raw_result)):
            if i % 1000 == 0: print(i)

            # for raw_result_input
            # each row: [0, 0, ..., 1, ..., 0 / 0, 0, ..., 1, ..., 0]
            input_row = []
            for j in range(2*N): input_row.append(0)
            input_row[raw_result[i][0] - N_start] = 1
            input_row[N + raw_result[i][1] - N_start] = 1

            # for raw_result_output
            output_row = [raw_result[i][2]]

            # append to raw_result_in/output
            raw_result_input.append(input_row)
            raw_result_output.append(output_row)

        RD.saveArray('raw_result_input.txt', raw_result_input, '\t', 500)
        RD.saveArray('raw_result_output.txt', raw_result_output, '\t', 500)

    # execute deep learning
    TRI = 'raw_result_input.txt'
    TRO = 'raw_result_output.txt'
    TEI = None
    TEO = None

    TE_real = None
    TE_report = None
    VAL_rate = 0.05
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    DL.deepLearning(TRI, TRO, TEI, TEO,
                    TE_real, TE_report, VAL_rate, VAL_report, modelConfig,
                    deviceName, epoch, printed, 'model')
