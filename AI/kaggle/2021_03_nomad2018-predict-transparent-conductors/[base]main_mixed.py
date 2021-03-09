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
from catboost import CatBoostRegressor

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

### WRITE RESULT (COMMON) ###
def writeResult(predict_tv, tv_output, VAL_rate, num, modelName):
    
    # write result
    result = []
    for i in range(len(predict_tv)):
        result.append([predict_tv[i]])

    if VAL_rate > 0:
        RD.saveArray(modelName + '_val_result_' + str(num) + '.txt', result, '\t', 500)
    else:
        RD.saveArray(modelName + '_test_result_' + str(num) + '.txt', result, '\t', 500)

    # validation mode -> compute RMSLE error
    # saved results are still NORMALIZED values
    if VAL_rate > 0:

        # convert prediction
        tv_output_ = []
        predict_tv_ = []

        # for tv_output
        for i in range(len(tv_output[0])):

            if num == 0: # formation_energy_ev_natom
                tv_output_.append(float(tv_output[0][i]) * 0.104078 + 0.187614)

            else: # bandgap_energy_ev
                tv_output_.append(float(tv_output[0][i]) * 1.006635 + 2.077205)

        # for predict_tv
        for i in range(len(predict_tv)):

            if num == 0: # formation_energy_ev_natom
                predict_tv_.append(float(predict_tv[i]) * 0.104078 + 0.187614)

            else: # bandgap_energy_ev
                predict_tv_.append(float(predict_tv[i]) * 1.006635 + 2.077205)

        # compute RMSLE and return
        print('\n\n ====[ ' + modelName + ' / valid=True ]====')
        return rmsle(tv_output_, predict_tv_)

    return 0

################################
##                            ##
##    model 00 : lightGBM     ##
##                            ##
################################
    
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# ref: https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y, y_pred):
    
    print('\ny: (30)')
    print(y[:30])
    print('\ny_pred: (30)')
    print(y_pred[:30])
    print('')
        
    assert(len(y) == len(y_pred))
	
    terms_to_sum = [(math.log(float(y_pred[i]) + 1) - math.log(float(y[i]) + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
    
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
              'learning_rate': 0.01,
              'feature_fraction': 0.93,
              'bagging_fraction': 0.93,
              'bagging_freq': 1,
              'metric': 'l2',
              'num_threads': 1,
              'seed': None}

    # create model
    if VAL_rate > 0:
        model = lgb.train(params, train_ds, 5000, test_ds, verbose_eval=50, early_stopping_rounds=200)
    else:
        model = lgb.train(params, train_ds, 5000, train_ds, verbose_eval=50, early_stopping_rounds=200)

    # predict
    predict_tv = model.predict(tv_input)

    # write result and return RMSLE error
    return writeResult(predict_tv, tv_output, VAL_rate, num, 'm00_lightGBM')

################################
##                            ##
##    model 01 : CatBoost     ##
##                            ##
################################

# ref: https://www.kaggle.com/tunguz/simple-catboost

def model_01_CatBoost(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report, num):

    # create Pandas DataFrame
    # tv_input  : test / validation input
    # tv_output : test / validation output
    (train_input, train_output, tv_input, tv_output) = create_dataframe(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report)

    model = CatBoostRegressor(iterations=4000,
                              learning_rate=0.01,
                              depth=4,
                              loss_function='RMSE',
                              eval_metric='RMSE',
                              random_seed=2021,
                              od_type='Iter',
                              od_wait=50)

    # create model
    if VAL_rate > 0:
        model.fit(train_input, train_output, eval_set=(tv_input, tv_output), use_best_model=True, verbose=False)
    else:
        model.fit(train_input, train_output, eval_set=None, use_best_model=True, verbose=False)

    # predict
    predict_tv = model.predict(tv_input)

    # write result and return RMSLE error
    return writeResult(predict_tv, tv_output, VAL_rate, num, 'm01_CatBoost')

################################
##                            ##
##           MAIN             ##
##                            ##
################################

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
    final_rmsles = [['LGBM', 'CATBOOST']]

    for num in [0, 1]:
        rmsles = []
            
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

        # print mode
        if VAL_rate > 0.0:
            print('VALIDATION mode')
        else:
            print('TEST mode')

        # training and test

        # model 00 : lightGBM
        rmse_00 = model_00_lightGBM(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report, num)
        rmsles.append(rmse_00)

        # model 01 : CatBoost
        rmse_01 = model_01_CatBoost(TRI_array, TRO_array, TEI_array, TEO, TE_report, VAL_rate, VAL_report, num)
        rmsles.append(rmse_01)

        # write FINAL ANSWER
        # in VALIDATION MODE, there will be an error
            
        # read file
        if VAL_rate > 0.0:
            testValResult_00 = RD.loadArray('m00_lightGBM_val_result_' + str(num) + '.txt')
            testValResult_01 = RD.loadArray('m01_CatBoost_val_result_' + str(num) + '.txt')
        else:
            testValResult_00 = RD.loadArray('m00_lightGBM_test_result_' + str(num) + '.txt')
            testValResult_01 = RD.loadArray('m01_CatBoost_test_result_' + str(num) + '.txt')

        # write final result
        finalResult_00 = []
        finalResult_01 = []
            
        for i in range(len(testValResult_00)):
                
            if num == 0: # formation_energy_ev_natom
                finalResult_00.append([float(testValResult_00[i][0]) * 0.104078 + 0.187614])
                finalResult_01.append([float(testValResult_01[i][0]) * 0.104078 + 0.187614])

            else: # bandgap_energy_ev
                finalResult_00.append([float(testValResult_00[i][0]) * 1.006635 + 2.077205])
                finalResult_01.append([float(testValResult_01[i][0]) * 1.006635 + 2.077205])

        # write file
        RD.saveArray('to_submit_' + str(num) + '_m00.txt', finalResult_00)
        RD.saveArray('to_submit_' + str(num) + '_m01.txt', finalResult_01)

        final_rmsles.append(rmsles)

    # append final rmsles
    methods = len(final_rmsles[0])
    avg_rmsles = []
    
    for i in range(methods):
        avg_rmsles.append((float(final_rmsles[1][i]) + float(final_rmsles[2][i])) / 2)
    
    final_rmsles.append(avg_rmsles)

    # save final rmsles
    RD.saveArray('final_rmsles.txt', np.array(final_rmsles).T)
