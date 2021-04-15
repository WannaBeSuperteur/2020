import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# main
if __name__ == '__main__':

    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TRI = 'train_input.txt'
    TRO = 'train_output'
    TEI = 'test_input.txt'
    TEO = 'test_output'

    TE_real = None
    TE_report = 'report_test'
    VAL_rate = 0.0
    VAL_report = 'report_val'
    modelConfig = 'model_config.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # training and test
    # 'config.txt' is used for configuration
    for i in ['funny', 'useful', 'cool']:
        DL.deepLearning(TRI, TRO + '_' + i + '.txt', TEI, TEO + '_' + i + '.txt',
                        TE_real, TE_report + '_' + i + '.txt',
                        VAL_rate, VAL_report + '_' + i + '.txt',
                        modelConfig, deviceName, epoch, printed, 'model_' + i)
