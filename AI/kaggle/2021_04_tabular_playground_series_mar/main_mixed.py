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

def create_dataframe(TRI_array, TRO_array, TEI_array,
                     TEO, TE_report, VAL_rate, VAL_report):

    # length
    train_rows = len(TRI_array)
    test_rows = len(TEI_array)
    train_cols = len(TRI_array[0])

    # TEST
    if VAL_rate == 0:

        # test : create dataframe
        TRI_df = pd.DataFrame(np.array(TRI_array).astype(float))
        TRO_df = pd.DataFrame(np.array(TRO_array).astype(float))
        TEI_df = pd.DataFrame(np.array(TEI_array).astype(float))

        return (TRI_df, TRO_df, TEI_df, None)

    # VALID
    else:

        # valid : preparation for the data
        (TRI_, TRO_, VAI_, VAO_) = designate_val(TRI_array, TRO_array, train_rows, VAL_rate)

        # valid : create dataframe
        TRI_df = pd.DataFrame(np.array(TRI_).astype(float))
        TRO_df = pd.DataFrame(np.array(TRO_).astype(float))
        VAI_df = pd.DataFrame(np.array(VAI_).astype(float))
        VAO_df = pd.DataFrame(np.array(VAO_).astype(float))

        return (TRI_df, TRO_df, VAI_df, VAO_df)

# reference: https://www.kaggle.com/gunesevitan/tabular-playground-series-jan-2021-models
#            http://machinelearningkorea.com

################################
##                            ##
##    model 00 : lightGBM     ##
##                            ##
################################
def model_00_lightGBM(TRI_array, TRO_array, TEI_array,
                      TEO, TE_report, VAL_rate, VAL_report):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input, tv_output) = create_dataframe(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report)

    # convert to lightgbm dataset
    train_ds = lgb.Dataset(train_input, label=train_output)
    test_ds = lgb.Dataset(tv_input, label=tv_output)

    # set parameters
    # refer to https://www.kaggle.com/awwalmalhi/extreme-fine-tuning-lgbm-using-7-step-training (0.84193)
    params = {'learning_rate': 0.01,
              'subsample': 0.8,
              'colsample_bytree': 0.21,
              'reg_lambda': 11,
              'reg_alpha': 17,
              'min_child_samples': 31,
              'max_depth': 16,
              'boosting': 'gbdt',
              'objective': 'regression',
              'metric': 'rmse',
              'is_training_metric': True,
              'num_leaves': 64,
              'max_bin': 512,
              'cat_smooth': 80,
              'cat_l2': 0.03,
              'bagging_freq': 3,
              'seed': None}

    # create model
    if VAL_rate > 0:
        model = lgb.train(params, train_ds, 2200, test_ds, verbose_eval=30, early_stopping_rounds=200)
    else:
        model = lgb.train(params, train_ds, 4000, train_ds, verbose_eval=40, early_stopping_rounds=200)

    # predict
    predict_train = model.predict(train_input)
    predict_tv = model.predict(tv_input)

    # write result
    result = []
    for i in range(len(predict_tv)):
        result.append([predict_tv[i]])

    if VAL_rate > 0:
        RD.saveArray('m00_lightGBM_val_result.txt', result, '\t', 500)
    else:
        RD.saveArray('m00_lightGBM_test_result.txt', result, '\t', 500)

    # validation mode -> compute RMSE error
    if VAL_rate > 0:
        rmse = math.sqrt(mean_squared_error(tv_output, predict_tv))
        roc_auc = roc_auc_score(tv_output, predict_tv)
        return (rmse, roc_auc)

    return (0, 0)

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    
    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TRI = 'train_input.txt'
    TRO = 'train_output.txt'
    TEI = 'test_input.txt'
    TEO = ['test_output.txt']

    TE_real = None
    TE_report = 'report_test.txt'
    VAL_rate = 0.0
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

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

    final_metrics = []

    # training and test

    # model 00 : lightGBM
    (rmse_00, rocauc_00) = model_00_lightGBM(TRI_array, TRO_array, TEI_array,
                                             TEO, TE_report, VAL_rate, VAL_report)

    final_metrics.append([rmse_00, rocauc_00])

    # write final result (RMSE and ROC-AUC)
    if VAL_rate > 0.0:
        RD.saveArray('final_metrics.txt', final_metrics, '\t', 500)

    # write FINAL ANSWER
    # in VALIDATION MODE, there will be an error
    
    # read file
    testResult = RD.loadArray('m00_lightGBM_test_result.txt')

    # write final result
    finalResult = []
    
    for i in range(len(testResult)):
        finalResult.append([float(testResult[i][0])])

    # write file
    RD.saveArray('to_submit.txt', finalResult)
