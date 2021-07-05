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

import lightgbm
from lightgbm import LGBMClassifier

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

# lightGBM (ref: Daniel's Kaggle - https://www.kaggle.com/danieljhk/lgbm-baseline/output?scriptVersionId=67233275)
def getLGBMModel():

    # LGBM classifier
    params = {
        'learning_rate': 0.22296318552587047,
        'max_depth': 3,
        'num_leaves': 85,
        'n_estimators': 253,
        'min_child_samples': 115,
        'colsample_bytree': 0.8217788951229221,
        'subsample': 0.7488378856367371,
        'reg_lambda': 731.663645944627,
        'reg_alpha': 0.10179189160953969,
        'scale_pos_weight': 0.18191544558601597,
        'class_weight': None,
        'objective': 'multiclass',
        'random_state': 314,
        'verbose': -1
    }
    model = LGBMClassifier(**params)

    return model

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName):
    print(' ==== predict using model : ' + str(model) + ' ====')
                            
    model.fit(train_X, train_Y, verbose=1000)
    
    prediction = model.predict_proba(test_X)
    prediction = np.clip(prediction, 0.08, 0.95)
    prediction = pd.DataFrame(prediction)

    # save prediction
    if predictionFileName != None:
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
    model0 = getCatBoostModel() # catboost classifier
    model1 = getLGBMModel() # lightGBM classifier

    # array of models
    modelNames = ['catboost', 'lightgbm']
    models = [model0, model1]

    # using weighted average of models
    for i in range(len(weights)):
        predictionUsingModel = weights[i] * predict(models[i], train_X, train_Y, test_X, None)
        print('\nprediction using model ' + str(i) + ' : (w=' + str(weights[i]) + ') = \n' + str(predictionUsingModel[:5]) + '\n')
        
        if i == 0:
            prediction = predictionUsingModel
        else:
            prediction += predictionUsingModel

    # save prediction and weights for each model
    prediction.to_csv(predictionFileName + '.csv')

    return prediction

# run with normalize / log2 option for INPUT data (True/False each)

# noramlize : noramlize X using average and stddev
# log2      : X -> log2(X+1) if X >= 0
#                  0         if X <  0
def run(weights, train_df, test_df, dic, normalize, log2, final, fileID):

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
    if final == False:

        k = 5
        unit_rows = int(train_rows / k) # rows in a k-folded unit
        error = []

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
            valid_prediction = predictWithModels(weights, train_train_X, train_train_Y, train_valid_X, 'val' + str(fileID) + '_' + str(i))
            error.append(round(computeMulticlassLoss(valid_prediction, train_valid_Y), 6))

        print('loss = ' + str(error))
        pd.DataFrame(weights).to_csv('val' + str(fileID) + '_weights.csv')
        pd.DataFrame(error).to_csv('val' + str(fileID) + '_loss.csv')

        return error

    # final prediction
    else: # final == True
        predictWithModels(weights, train_X, train_Y, test_X, 'fin' + str(fileID))

if __name__ == '__main__':

    # read data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')

    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3}

    # run for 200 rounds
    rounds = 5 # temp

    # initial weights for each model
    weights = [0.5, 0.5]

    # log
    log = ''

    for i in range(rounds):
        error = run(weights, train_df, test_df, dic, False, False, False, 10000 + i*10)
        meanError = np.mean(error)
        log += ('[ round ' + str(i) + ' ]\nweights=' + str(weights) + ' error=' + str(error) + ' avg=' + str(round(meanError, 6)) + '\n')

        # explore neighboring cases
        weights_neighbor0 = [min(weights[0] + 0.05, 1.0), max(weights[1] - 0.05, 0.0)]
        weights_neighbor1 = [max(weights[0] - 0.05, 0.0), min(weights[1] + 0.05, 1.0)]

        error_neighbor0 = run(weights_neighbor0, train_df, test_df, dic, False, False, False, 10000 + i*10 + 1)
        error_neighbor1 = run(weights_neighbor1, train_df, test_df, dic, False, False, False, 10000 + i*10 + 2)

        meanError_neighbor0 = np.mean(error_neighbor0)
        meanError_neighbor1 = np.mean(error_neighbor1)

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using meanError
        log += ('neighbor0=' + str(weights_neighbor0) + ' error=' + str(meanError_neighbor0) + '\n')
        log += ('neighbor1=' + str(weights_neighbor1) + ' error=' + str(meanError_neighbor1) + '\n')

        # error(neighbor1) < error < error(neighbor0) -> move to neighbor1
        if meanError < meanError_neighbor0 and meanError > meanError_neighbor1:
            weights[0] = max(weights[0] - 0.05, 0.0)
            weights[1] = min(weights[1] + 0.05, 1.0)
            log += 'move to neighbor0\n'

        # error(neighbor0) < error < error(neighbor1) -> move to neighbor0
        elif meanError > meanError_neighbor0 and meanError < meanError_neighbor1:
            weights[0] = min(weights[0] + 0.05, 1.0)
            weights[1] = max(weights[1] - 0.05, 0.0)
            log += 'move to neighbor1\n'

        else:
            break

    # save log for each round at finish
    f = open('log.txt', 'w')
    f.write(log)
    f.close()

    # final prediction
    run(weights, train_df, test_df, dic, False, False, True, 99999)

    # save log for final prediction
    log += ('final prediction with ' + str(weights))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
