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

# designate data to validate
def designate_val(TRI_array, TRO_array, train_rows, VAL_rate):

    # designate rows to validate
    validArray = []
    for i in range(train_rows): validArray.append(0)

    count = 0 # count of validate-designated rows

    while count < train_rows * VAL_rate:
        num = random.randint(0, train_rows - 1)
        
        if validArray[num] == 0:
            validArray[num] = 1
            count += 1

    # return training and validation data
    TRI_ = [] # training input
    TRO_ = [] # training output
    VAI_ = [] # validation input
    VAO_ = [] # validation output

    for i in range(train_rows):

        # for training rows
        if validArray[i] == 0:
            TRI_.append(TRI_array[i])
            TRO_.append(TRO_array[i])

        # for validation data
        else:
            VAI_.append(TRI_array[i])
            VAO_.append(TRO_array[i])

    return (TRI_, TRO_, VAI_, VAO_)

# create dataframe
def create_dataframe(TRI_array, TRO_array, TEI_array):

    TRI_df = pd.DataFrame(np.array(TRI_array).astype(float))
    TRO_df = pd.DataFrame(np.array(TRO_array).astype(float))
    TEI_df = pd.DataFrame(np.array(TEI_array).astype(float))

    return (TRI_df, TRO_df, TEI_df)

# reference: https://www.kaggle.com/gomes555/tps-mar2021-lightgbm-optuna-opt-data-prep
#            http://machinelearningkorea.com

################################
##                            ##
##    model 00 : lightGBM     ##
##                            ##
################################
def lightGBM(TRI_array, TRO_array, TEI_array, count):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input) = create_dataframe(TRI_array, TRO_array, TEI_array)

    # convert to lightgbm dataset
    train_ds = lgb.Dataset(train_input, label=train_output)

    # set parameters
    # refer to https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble (0.81722)
    params = {'metric': 'binary_logloss',
              'objective': 'binary',
              'random_state': 2021 + count, # SEED = 2021...
              'learning_rate': 0.01,
              'min_child_samples': 150,
              'reg_alpha': 3e-5,
              'reg_lambda': 9e-2,
              'num_leaves': 20,
              'max_depth': 16,
              'colsample_bytree': 0.8,
              'subsample': 0.8,
              'subsample_freq': 2,
              'max_bin': 240}

    # create model
    model = lgb.train(params, train_ds, 2000, train_ds, verbose_eval=20, early_stopping_rounds=200)

    # predict
    predict_tv = model.predict(tv_input)
    predictions = len(predict_tv)

    RD.saveArray('lightGBM_tv_predict_' + str(count) + '.txt', np.array([predict_tv]).T)

################################
##                            ##
##  model 01 : DecisionTree   ##
##                            ##
################################
def DecisionTree(TRI_array, TRO_array, TEI_array, count):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input) = create_dataframe(TRI_array, TRO_array, TEI_array)

    # convert to lightgbm dataset
    train_ds = lgb.Dataset(train_input, label=train_output)

    # set parameters and create model
    # refer to https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble (0.81722)
    model = DecisionTreeClassifier(
        max_depth = 3 + count % 2,
        min_samples_leaf = 2 + count // 2,
        random_state = 2021 + count
    )
    model.fit(train_input, train_output)

    # predict
    predict_tv = model.predict(tv_input)
    predictions = len(predict_tv)

    RD.saveArray('DecisionTree_tv_predict_' + str(count) + '.txt', np.array([predict_tv]).T)

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    
    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TE_real = None
    TE_report = 'report_test.txt'
    VAL_rate = 0.0
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

    if VAL_rate > 0: # valid mode
        TRI = 'train_train_input.txt'
        TRO = 'train_train_output.txt'
        TEI = 'train_valid_input.txt'
        TEO = 'train_valid_predict.txt'
    else: # test mode
        TRI = 'train_input.txt'
        TRO = 'train_output.txt'
        TEI = 'test_input.txt'
        TEO = 'test_predict.txt'

    # load array
    print('loading training input...')
    TRI_array = RD.loadArray(TRI, '\t', UTF8=False, type_='f')
    
    print('loading training output...')
    TRO_array = RD.loadArray(TRO, '\t', UTF8=False, type_='f')
    
    print('loading test input...')
    TEI_array = RD.loadArray(TEI, '\t', UTF8=False, type_='f')

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # print mode
    if VAL_rate > 0.0:
        print('VALIDATION mode')
    else:
        print('TEST mode')

    times = 4
    algorithm = 'lightGBM DecisionTree'

    # training and test
    # 'config.txt' is used for configuration
    for i in range(times):

        # algorithm 0: basic deep learning
        if 'deepLearning' in algorithm:
            
            DL.deepLearning(TRI, TRO, TEI, TEO[:-4] + '_' + str(i) + '.txt',
                            TE_real, TE_report[:-4] + '_' + str(i) + '.txt',
                            0, VAL_report[:-4] + '_' + str(i) + '.txt',
                            modelConfig, deviceName, epoch, printed, 'model_' + str(i))

        # algorithm 1: lightGBM
        if 'lightGBM' in algorithm:

            # load array
            print('loading training input...')
            TRI_array = RD.loadArray(TRI, '\t', UTF8=False, type_='f')
    
            print('loading training output...')
            TRO_array = RD.loadArray(TRO, '\t', UTF8=False, type_='f')
    
            print('loading test input...')
            TEI_array = RD.loadArray(TEI, '\t', UTF8=False, type_='f')

            # execute lightGBM
            lightGBM(TRI_array, TRO_array, TEI_array, i)

        # algorithm 2: DecisionTree
        if 'DecisionTree' in algorithm:
            
            # load array
            print('loading training input...')
            TRI_array = RD.loadArray(TRI, '\t', UTF8=False, type_='f')
    
            print('loading training output...')
            TRO_array = RD.loadArray(TRO, '\t', UTF8=False, type_='f')
    
            print('loading test input...')
            TEI_array = RD.loadArray(TEI, '\t', UTF8=False, type_='f')

            # execute lightGBM
            DecisionTree(TRI_array, TRO_array, TEI_array, i)
