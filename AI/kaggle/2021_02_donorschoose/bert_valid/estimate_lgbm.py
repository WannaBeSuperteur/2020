# train_train.csv, train_test.csv -> bert_train_result.txt, bert_test_result.txt
# additional need: bert_max_lengths_train.txt

# ref: https://insightcampus.co.kr/insightcommunity/?mod=document&uid=12854
#      https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
#      https://www.kaggle.com/linwenxing/glove-vector-lstm

import sys
sys.path.insert(0, '../../AI_BASE')
import my_extract as ME

import math
import numpy as np
import readData as RD
import deepLearning_main as DL
import pandas as pd
import random

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# create dataframe
def create_dataframe(TRI_array, TRO_array, TEI_array, TEO_array=None):

    TRI_df = pd.DataFrame(np.array(TRI_array).astype(float))
    TRO_df = pd.DataFrame(np.array(TRO_array).astype(float))
    TEI_df = pd.DataFrame(np.array(TEI_array).astype(float))

    if TEO_array != None:
        TEO_df = pd.DataFrame(np.array(TEO_array).astype(float))
        return (TRI_df, TRO_df, TEI_df, TEO_df)
    else:
        return (TRI_df, TRO_df, TEI_df)

# use lightGBM
def lightGBM(TRI_array, TRO_array, TEI_array, TEO_array, count, valid):

    # create Pandas DataFrame and convert to lightgbm dataset
    # tv_input  : test / validation input
    # tv_output : test / validation output
    if valid == True:
        (train_input, train_output, tv_input, tv_output) = create_dataframe(TRI_array, TRO_array, TEI_array, TEO_array)
        train_ds = lgb.Dataset(train_input, label=train_output)
        valid_ds = lgb.Dataset(tv_input, label=tv_output)

    else:
        (train_input, train_output, tv_input) = create_dataframe(TRI_array, TRO_array, TEI_array)
        train_ds = lgb.Dataset(train_input, label=train_output)

    # set parameters
    params = {'metric': 'AUC',
              'objective': 'regression',
              'random_state': 2021 + count, # SEED = 2021...
              'learning_rate': 0.005,
              'min_child_samples': 256,
              'reg_alpha': 3e-5,
              'reg_lambda': 9e-2,
              'num_leaves': 32,
              'max_depth': 32,
              'colsample_bytree': 0.8,
              'subsample': 0.8,
              'subsample_freq': 2,
              'max_bin': 1024}

    # create model
    if valid == True:
        model = lgb.train(params, train_ds, 4000, valid_ds, verbose_eval=30, early_stopping_rounds=200)
    else:
        model = lgb.train(params, train_ds, 4000, train_ds, verbose_eval=30, early_stopping_rounds=200)

    # predict
    predict_tv = model.predict(tv_input)
    predictions = len(predict_tv)

    if valid == True:
        RD.saveArray('lightGBM_tv_valid_' + str(count) + '.txt', np.array([predict_tv]).T)
    else:
        RD.saveArray('lightGBM_tv_predict_' + str(count) + '.txt', np.array([predict_tv]).T)

# designate rows to valid
def designateValidRows(rows, validRate):
    
    isValidRow = []
    for i in range(rows): isValidRow.append(False)
    valid_count = 0

    while valid_count < rows * validRate:
        row = random.randint(0, rows-1)
            
        if isValidRow[row] == False:
            isValidRow[row] = True
            valid_count += 1

    return isValidRow

# main function
def mainFunc(times):

    # define configuration
    trainFile = 'train.csv'
    testFile = 'test.csv'
    
    trainExtractedFile = 'train_extracted_lgbm.txt'
    testExtractedFile = 'test_extracted_lgbm.txt'
    
    trainTrainInputFile = 'train_train_input_lgbm.txt'
    trainTrainOutputFile = 'train_train_output_lgbm.txt'
    trainValidInputFile = 'train_valid_input_lgbm.txt'
    trainValidOutputFile = 'train_valid_output_lgbm.txt'
    
    valid = True
    validRate = 0.1

    # extract data from training dataset
    train_info = []
    test_info = []

    (option, title) = ME.getOptionsAndTitle()
    wordCount = ME.getWordCount(option, trainFile)

    # load training output data
    train_data = np.array(pd.read_csv(trainFile, dtype={11:str, 12:str}))
    train_output = train_data[:, 15:16].astype(float)

    # load data, if failed, save data
    try:
        train_input = RD.loadArray(trainExtractedFile, '\t')
        test_input = RD.loadArray(testExtractedFile, '\t')
        
    except:
        (train_input, _, onehot) = ME.extract(trainFile, option, title, wordCount, None, [2016, 2017])
        (test_input, _, _) = ME.extract(testFile, option, title, wordCount, onehot, [2016, 2017])

        RD.saveArray(trainExtractedFile, train_input, '\t', 500)
        RD.saveArray(testExtractedFile, test_input, '\t', 500)

    # separate valid dataset if valid==True
    if valid == True:

        rows = len(train_input)
        
        train_train_input = []
        train_train_output = []
        train_valid_input = []
        train_valid_output = []

        # designate rows to valid
        isValidRow = designateValidRows(rows, validRate)
        
        # prepare train and valid dataset
        for i in range(len(train_input)):  
            if isValidRow[i]:
                train_valid_input.append(train_input[i])
                train_valid_output.append(train_output[i])
            else:
                train_train_input.append(train_input[i])
                train_train_output.append(train_output[i])

        # save files
        RD.saveArray(trainTrainInputFile, train_train_input, '\t', 500)
        RD.saveArray(trainTrainOutputFile, train_train_output, '\t', 500)
        RD.saveArray(trainValidInputFile, train_valid_input, '\t', 500)
        RD.saveArray(trainValidOutputFile, train_valid_output, '\t', 500)

    # print dataset info
    print('\ntrain_input:')
    print(np.shape(train_input))
    print(np.array(train_input))

    print('\ntrain_output:')
    print(np.shape(train_output))
    print(np.array(train_output))

    print('\ntest_input:')
    print(np.shape(test_input))
    print(np.array(test_input))

    # predict using lightGBM
    for count in range(times):
        if valid == True:
            lightGBM(train_train_input, train_train_output,
                     train_valid_input, train_valid_output, count, True)
        else:
            lightGBM(train_input, train_output, test_input, None, count, False)

if __name__ == '__main__':

    # repeat 4 times
    mainFunc(4)
