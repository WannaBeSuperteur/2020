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
from sklearn.metrics import mean_squared_error, r2_score

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

################################
##                            ##
##    model 00 : lightGBM     ##
##                            ##
################################
def model_00_lightGBM(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report, num):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input, tv_output) = create_dataframe(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report)

    # convert to lightgbm dataset
    train_ds = lgb.Dataset(train_input, label=train_output)
    test_ds = lgb.Dataset(tv_input, label=tv_output)

    # set parameters
    # refer to https://www.kaggle.com/janpreets/using-the-atomic-coordinates-for-prediction (Private: 0.6988, Public: 0.5046)
    params = {'num_leaves': 8,
              'objective': 'regression',
              'min_data_in_leaf': 18,
              'learning_rate': 0.05,
              'feature_fraction': 0.93,
              'bagging_fraction': 0.93,
              'bagging_freq': 1,
              'metric': 'l2',
              'num_threads': 1,
              'seed': None}

    # create model
    if VAL_rate > 0:
        model = lgb.train(params, train_ds, 1500, test_ds, verbose_eval=25, early_stopping_rounds=200)
    else:
        model = lgb.train(params, train_ds, 1500, train_ds, verbose_eval=25, early_stopping_rounds=200)

    # predict
    predict_train = model.predict(train_input)
    predict_tv = model.predict(tv_input)

    # write result
    result = []
    for i in range(len(predict_tv)):
        result.append([predict_tv[i]])

    if VAL_rate > 0:
        RD.saveArray('m00_lightGBM_val_result_' + str(num) + '.txt', result, '\t', 500)
    else:
        RD.saveArray('m00_lightGBM_test_result_' + str(num) + '.txt', result, '\t', 500)

    # validation mode -> compute RMSE error
    if VAL_rate > 0:
        rmse = math.sqrt(mean_squared_error(tv_output, predict_tv))
        return rmse

    return 0

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    
    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column
    # removed modelConfig

    # meta info
    TRI = 'train_input.txt'
    TEI = 'test_input.txt'

    for num in [0, 1]:
        TRO = 'train_output_' + str(num) + '.txt'
        TEO = ['test_output_' + str(num) + '.txt']

        TE_real = None
        TE_report = 'report_test_' + str(num) + '.txt'
        VAL_rate = 0.0
        VAL_report = 'report_val_' + str(num) + '.txt'

        # load array
        TRI_array = RD.loadArray(TRI, '\t')
        TRO_array = RD.loadArray(TRO, '\t')
        TEI_array = RD.loadArray(TEI, '\t')

        # user data
        deviceName = input('device name (for example, cpu:0 or gpu:0)')
        epoch = int(input('epoch'))
        printed = int(input('printed? (0 -> do not print)'))

        # print mode
        if VAL_rate > 0.0:
            print('VALIDATION mode')
        else:
            print('TEST mode')

        final_rmses = []

        # training and test

        # model 00 : lightGBM
        rmse_00 = model_00_lightGBM(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report, num)

        final_rmses.append([rmse_00])

        # write final result (RMSE)
        if VAL_rate > 0.0:
            RD.saveArray('final_RMSE_' + str(num) + '.txt', final_rmses, '\t', 500)

        # write FINAL ANSWER
        # in VALIDATION MODE, there will be an error
        
        # read file
        testResult = RD.loadArray('m00_lightGBM_test_result_' + str(num) + '.txt')

        # write final result
        finalResult = []
        
        for i in range(len(testResult)):
            
            if num == 0: # formation_energy_ev_natom
                finalResult.append([float(testResult[i][0]) * 0.104078 + 0.187614])

            else: # bandgap_energy_ev
                finalResult.append([float(testResult[i][0]) * 1.006635 + 2.077205])

        # write file
        RD.saveArray('to_submit_' + str(num) + '.txt', finalResult)
