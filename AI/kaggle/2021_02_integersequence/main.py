import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL

# train
def train(num, deviceName, epoch, printed):

    # meta info
    TRI = 'input_' + str(num) + '.txt'
    TRO = 'output_' + str(num) + '.txt'
    TEI = None
    TEO = None

    TE_real = None
    TE_report = None
    VAL_rate = 0.05
    VAL_report = 'report_val_' + str(num) + '.txt'
    modelConfig = 'model_config_' + str(num) + '.txt'

    # training and test
    # 'config.txt' is used for configuration
    DL.deepLearning(TRI, TRO, TEI, TEO,
                    TE_real, TE_report, VAL_rate, VAL_report, modelConfig,
                    deviceName, epoch, printed, 'model_' + str(num) + '_' )

# main
if __name__ == '__main__':
    #train(0, 'cpu:0', 15, 0)
    #train(1, 'cpu:0', 15, 0)
    train(2, 'cpu:0', 15, 0)
    train(3, 'cpu:0', 15, 0)
    
