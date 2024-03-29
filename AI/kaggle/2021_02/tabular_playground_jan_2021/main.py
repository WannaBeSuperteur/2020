import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# main
if __name__ == '__main__':

    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column
    #
    # training output:
    # (original output value - 8.0)
    #
    # test output:
    # (converted output value + 8.0)

    # meta info
    TRI = 'train_input.txt'
    TRO = 'train_output.txt'
    TEI = 'test_input.txt'
    TEO = 'test_output.txt'

    TE_real = None
    TE_report = 'report_test.txt'
    VAL_rate = 0.05
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # training and test
    # 'config.txt' is used for configuration
    DL.deepLearning(TRI, TRO, TEI, TEO,
                    TE_real, TE_report, VAL_rate, VAL_report, modelConfig,
                    deviceName, epoch, printed, 'model')
