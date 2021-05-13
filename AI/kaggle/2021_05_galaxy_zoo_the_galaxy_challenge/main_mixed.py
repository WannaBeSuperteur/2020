import pandas as pd
import numpy as np
import math
import random

import sys
sys.path.insert(0, '../../AI_BASE')
import numpy as np
import readData as RD
import deepLearning_main as DL

import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    
    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TE_real = None
    TE_report = 'report_test.txt'
    VAL_rate = float(RD.loadArray('val_rate.txt')[0][0])
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

    TRI = 'train_input.txt'
    TRO = 'train_output.txt'
    TEI = 'test_input.txt'
    TEO = 'test_predict.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epochs'))
    printed = int(input('printed? (0 -> do not print)'))

    # print mode
    if VAL_rate > 0:
        print('VALIDATION mode')
    else:
        print('TEST mode')

    times = 1
    algorithm = 'deepLearning'

    # training and test
    # 'config.txt' is used for configuration
    for i in range(times):

        # load array
        print('loading training input...')
        TRI_array = RD.loadArray(TRI, '\t', UTF8=False, type_='f')
    
        print('loading training output...')
        TRO_array = RD.loadArray(TRO, '\t', UTF8=False, type_='f')
    
        print('loading test input...')
        TEI_array = RD.loadArray(TEI, '\t', UTF8=False, type_='f')

        # using basic deep learning
        DL.deepLearning(TRI, TRO, TEI, TEO[:-4] + '_' + str(i) + '.txt',
                        TE_real, TE_report[:-4] + '_' + str(i) + '.txt',
                        VAL_rate, VAL_report[:-4] + '_' + str(i) + '.txt',
                        modelConfig, deviceName, epoch, printed, 'model_' + str(i))
