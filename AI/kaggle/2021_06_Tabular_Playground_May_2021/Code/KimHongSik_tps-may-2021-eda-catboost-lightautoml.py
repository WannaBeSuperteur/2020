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

# predict and save the prediction
def predict(train_X, train_Y, test_X, predictionFileName):
    
    # catboost classifier
    model = CatBoostClassifier(iterations=17000,
                            learning_rate=0.01,
                            depth=4,
                            loss_function='MultiClass',
                            od_wait=1000,
                            od_type='Iter',                           
                            min_data_in_leaf=1,
                            max_ctr_complexity=15)                            
                            
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

# validate and return multiclass loss
# ground_truth : label      , shape=(rows, 1)
# prediction   : probability, shape=(rows, numClass)
def validate(prediction, ground_truth):

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

# run with normalize / log2 option for INPUT data (True/False each)

# noramlize : noramlize X using average and stddev
# log2      : X -> log2(X+1) if X >= 0
#                  0         if X <  0
def run(train_df, test_df, dic, normalize, log2, final, fileID):
    
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

    # split training data into train_train and train_valid data
    # training using train_train data
    # valid    using train_valid data
    train_rate = 0.9
    train_train_rows = train_rows * train_rate
    
    train_train_X = train_X.loc[:train_train_rows-1, :]
    train_train_Y = train_Y.loc[:train_train_rows-1, :]
    train_valid_X = train_X.loc[train_train_rows:, :]
    train_valid_Y = train_Y.loc[train_train_rows:, :]

    print(' ======== normalize:' + str(normalize) + ' log2:' + str(log2) + ' ========')
    print(train_train_X)
    print(train_train_Y)
    print(train_valid_X)
    print(train_valid_Y)
    print(' ========================================')

    # validation
    valid_prediction = predict(train_train_X, train_train_Y, train_valid_X, 'validation' + str(fileID))
    error = validate(valid_prediction, train_valid_Y)

    print('loss = ' + str(error))

    # final prediction
    if final == True:
        predict(train_X, train_Y, test_X, 'final_prediction' + str(fileID))

    return error

if __name__ == '__main__':

    train_rows = 100000
    numClass = 4

    # read data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')

    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3}

    # run
    error0 = run(train_df, test_df, dic, False, False, False, 0)
    error1 = run(train_df, test_df, dic, False, True, False, 1)
    error2 = run(train_df, test_df, dic, True, False, False, 2)
    error3 = run(train_df, test_df, dic, True, True, False, 3)

    print('normalize = False, log2 = False : ' + str(error0))
    print('normalize = False, log2 = True  : ' + str(error1))
    print('normalize = True , log2 = False : ' + str(error2))
    print('normalize = True , log2 = True  : ' + str(error3))
