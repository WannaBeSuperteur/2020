import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL
import lightgbm as lgb
import pandas as pd
import random

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

def create_dataframe(TRI_array, TRO_array, TEI_array, TEO, VAL_rate):

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

# reference: https://www.kaggle.com/gomes555/tps-mar2021-lightgbm-optuna-opt-data-prep
#            http://machinelearningkorea.com
#
# lightGBM
def model_00_lightGBM(TRI_array, TRO_array, TEI_array, TEO, VAL_rate, count):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input, tv_output) = create_dataframe(TRI_array, TRO_array, TEI_array, TEO, VAL_rate)

    # convert to lightgbm dataset
    train_ds = lgb.Dataset(train_input, label=train_output)
    test_ds = lgb.Dataset(tv_input, label=tv_output)

    # set parameters
    # refer to https://www.kaggle.com/gomes555/tps-mar2021-lightgbm-optuna-opt-data-prep (0.89765)
    params = {'resample': None,
              'boosting': 'gbdt',
              'num_leaves': 200,
              'min_child_samples': 32,
              'max_depth': 16,
              'max_delta_step': 8,
              'reg_alpha': 0.086,
              'reg_lambda': 9.52,
              'colsample_bytree': 0.35,
              'cat_smooth': 82,
              'cat_l2': 12,
              'learning_rate': 0.005, # modified
              'metric': 'rmse'}

    # create model
    if VAL_rate > 0:
        model = lgb.train(params, train_ds, 3500, test_ds, verbose_eval=15, early_stopping_rounds=200)
    else:
        model = lgb.train(params, train_ds, 6000, train_ds, verbose_eval=20, early_stopping_rounds=200)

    # predict
    predict_train = model.predict(train_input)
    predict_tv = model.predict(tv_input)

    # write result
    result = []
    for i in range(len(predict_tv)):
        result.append([predict_tv[i]])

    if VAL_rate > 0:
        RD.saveArray('lightGBM_val_result_' + str(count) + '.txt', result, '\t', 500)
    else:
        RD.saveArray('lightGBM_test_result_' + str(count) + '.txt', result, '\t', 500)

    # validation mode -> compute RMSE error
    if VAL_rate > 0:
        rmse = math.sqrt(mean_squared_error(tv_output, predict_tv))
        roc_auc = roc_auc_score(tv_output, predict_tv)
        return (rmse, roc_auc)

    # test mode -> return zero tuple
    return (0, 0)

# main
if __name__ == '__main__':

    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TRI = 'train_input.txt'
    TRO = 'train_output.txt'
    TEI = 'test_input.txt'
    TEO = 'test_output.txt'

    TE_real = None
    TE_report = 'report_test.txt'
    TE_report_lgbm = 'report_test_lgbm.txt'
    
    VAL_rate = 0.0
    VAL_report = 'report_val.txt'
    VAL_report_lgbm = 'report_val_lgbm.txt'
    
    modelConfig = 'model_config.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    times = 1
    algorithm = 'lightGBM'

    # training and test
    # 'config.txt' is used for configuration
    for i in range(times):

        # algorithm 0: basic deep learning
        if algorithm == 'deepLearning':
            
            DL.deepLearning(TRI, TRO, TEI, TEO[:-4] + '_' + str(i) + '.txt',
                            TE_real, TE_report[:-4] + '_' + str(i) + '.txt',
                            VAL_rate, VAL_report[:-4] + '_' + str(i) + '.txt',
                            modelConfig, deviceName, epoch, printed, 'model_' + str(i))

        # algorithm 1: lightGBM
        elif algorithm == 'lightGBM':

            # load array
            print('loading training input...')
            TRI_array = RD.loadArray(TRI, '\t', UTF8=False, type_='f')
    
            print('loading training output...')
            TRO_array = RD.loadArray(TRO, '\t', UTF8=False, type_='f')
    
            print('loading test input...')
            TEI_array = RD.loadArray(TEI, '\t', UTF8=False, type_='f')

            # execute lightGBM
            (rmse_00, rocauc_00) = model_00_lightGBM(TRI_array, TRO_array, TEI_array, TEO, VAL_rate, i)

            # write final result (RMSE and ROC-AUC)
            if VAL_rate > 0.0:
                RD.saveArray('final_metrics_lgbm_' + str(i) + '.txt', [[rmse_00, rocauc_00]], '\t', 500)
