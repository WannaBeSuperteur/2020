# using https://www.kaggle.com/mehrankazeminia/tps-may-21-catboost-tabularutilizedautoml

# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# team members' Python codes
import Daniel_BaselineModels as Daniel_Baseline

# models
# XGBoost            : An Jun Hang's idea
# LR, LDA, ..., LGBM : Daniel's idea
# CatBoost           : Hong Sik Kim's idea
def getCatBoostModel():

    # catboost classifier
    model = CatBoostClassifier(iterations=17000,
                            learning_rate=0.01,
                            depth=4,
                            loss_function='MultiClass',
                            od_wait=1000,
                            od_type='Iter',                           
                            min_data_in_leaf=1,
                            max_ctr_complexity=15)

    return model

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName):                            
                            
    model.fit(train_X, train_Y, verbose=1000)
    
    prediction = model.predict_proba(test_X)
    prediction = np.clip(prediction, 0.08, 0.95)
    prediction = pd.DataFrame(prediction)

    # save prediction
    prediction.to_csv(predictionFileName + '.csv')
    return prediction

# one-hot encode dataset (not used in this file)
def encodeOneHot(df, numClass):

    rows = len(df.index)
    encodedData = np.zeros((rows, numClass))

    for i in range(rows):
        encodedData[i][df.iloc[i, 0]] = 1

    encodedData = pd.DataFrame(encodedData)

    return encodedData

# compute and return multiclass loss
# ground_truth : label      , shape=(rows, 1)
# prediction   : probability, shape=(rows, numClass)
def computeMulticlassLoss(prediction, ground_truth):

    prediction_np = np.array(prediction)
    ground_truth_np = np.array(ground_truth)

    return log_loss(ground_truth, prediction)

# normalize DataFrame using average and stddev
def applyNormalization(df):
    return (df - df.mean()) / df.std()

# apply X -> log2(X+1) if X >= 0,
#            0         if X <  0
# to DataFrame
def applyLog(df):
    df[df >= 0] = pd.DataFrame(np.log2(df + 1))
    df[df < 0] = 0
    return df

# predict using multiple models
def predictWithModels(weights, train_X, train_Y, test_X, predictionFileName):

    # models (classifiers)
    model = getCatBoostModel() # catboost classifier

    # array of models
    models = [model]

    # using weighted average of models
    for i in range(len(weights)):
        if i == 0:
            prediction = weights[i] * predict(model, train_X, train_Y, test_X, predictionFileName)
        else:
            prediction += weights[i] * predict(model, train_X, train_Y, test_X, predictionFileName)

    return prediction

# run with normalize / log2 option for INPUT data (True/False each)

# noramlize : noramlize X using average and stddev
# log2      : X -> log2(X+1) if X >= 0
#                  0         if X <  0
def run(train_df, test_df, dic, normalize, log2, final, fileID):

    train_rows = 100000
    numClass = 4
    
    # extract training and test data
    train_X = train_df.loc[:, 'feature_0':'feature_49']
    train_Y = train_df.loc[:, 'target':]
    test_X = test_df.loc[:, 'feature_0':'feature_49']
    train_Y['target'].replace(dic, inplace=True)

    # apply X -> log2(X+1) to train_X and test_X
    if log2 == True:
        train_X = applyLog(train_X)
        test_X = applyLog(test_X)

    # noramlize train_X and test_X
    if normalize == True:
        train_X = applyNormalization(train_X)
        test_X = applyNormalization(test_X)

    # split training data into train_train and train_valid data using K-fold
    # training using train_train data
    # valid    using train_valid data

    k = 5
    unit_rows = int(train_rows / k) # rows in a k-folded unit
    error = []

    # weight
    weights = [1.0]

    for i in range(k):
    
        train_train_X = pd.concat([train_X.loc[:unit_rows*i - 1, :], train_X.loc[unit_rows*(i+1):, :]])
        train_train_Y = pd.concat([train_Y.loc[:unit_rows*i - 1, :], train_Y.loc[unit_rows*(i+1):, :]])
        train_valid_X = train_X.loc[unit_rows*i : unit_rows*(i+1) - 1, :]
        train_valid_Y = train_Y.loc[unit_rows*i : unit_rows*(i+1) - 1, :]

        print(' ======== normalize:' + str(normalize) + ' log2:' + str(log2) + ' ========')
        print(train_train_X)
        print(train_train_Y)
        print(train_valid_X)
        print(train_valid_Y)
        print(' ========================================')

        # validation
        valid_prediction = predictWithModels(weights, train_train_X, train_train_Y, train_valid_X, 'validation' + str(fileID) + '_' + str(i))
        error.append(round(computeMulticlassLoss(valid_prediction, train_valid_Y), 6))

    print('loss = ' + str(error))

    # final prediction
    if final == True:
        predictWithModels(weights, train_X, train_Y, test_X, 'final_prediction' + str(fileID) + '_' + str(i))

    return error

if __name__ == '__main__':

    # read data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')

    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3}

    # run
    error0 = run(train_df, test_df, dic, False, False, False, 0)
    error1 = run(train_df, test_df, dic, False, True, False, 1)
    error2 = run(train_df, test_df, dic, True, False, False, 2)
    error3 = run(train_df, test_df, dic, True, True, False, 3)

    print('\nnormalize = False, log2 = False :\n' + str(error0) + '\navg = ' + str(np.mean(error0)))
    print('\nnormalize = False, log2 = True  :\n' + str(error1) + '\navg = ' + str(np.mean(error1)))
    print('\nnormalize = True , log2 = False :\n' + str(error2) + '\navg = ' + str(np.mean(error2)))
    print('\nnormalize = True , log2 = True  :\n' + str(error3) + '\navg = ' + str(np.mean(error3)))
