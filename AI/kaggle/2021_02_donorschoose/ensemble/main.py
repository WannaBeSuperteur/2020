# use template of https://github.com/WannaBeSuperteur/2020/blob/master/AI/kaggle/2021_06_Tabular_Playground_May_2021/Code/KimHongSik_tps-may-2021-eda-catboost-lightautoml.py

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
from sklearn.metrics import roc_auc_score

import lightgbm
import xgboost
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# LGBM model
# ref : https://www.kaggle.com/boomberung/eda-tf-idf-lightgbm
def getLGBMModel(learning_rate):

    params = {
         'boosting_type': 'gbdt',
         'objective': 'binary',
         'metric': 'auc',
         'max_depth': 16,
         'num_leaves': 31,
         'learning_rate': learning_rate,
         'feature_fraction': 0.85,
         'bagging_fraction': 0.85,
         'bagging_freq': 5,
         'verbose': 1,
         'num_threads': 1,
         'lambda_l2': 1,
         'min_gain_to_split': 0,
    }
    
    model = LGBMClassifier(**params)
    return model

# ref : https://www.kaggle.com/safavieh/ultimate-feature-engineering-xgb-lgb-nn
def getXGBoostModel():

    params = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.25,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1,
    }

    model = XGBClassifier(**params)
    return model

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName):
    print(' ==== predict using model : ' + str(model) + ' ====')
                            
    model.fit(train_X, train_Y)

    prediction = model.predict_proba(test_X)
    prediction = pd.DataFrame(prediction).iloc[:, 1:2]

    # save prediction
    if predictionFileName != None:
        prediction.to_csv(predictionFileName + '.csv')
        
    return prediction

# compute and return ROC-AUC score
# ground_truth : label      , shape=(rows, 1)
# prediction   : probability, shape=(rows, numClass)
def computeROCAUC(prediction, ground_truth):

    prediction_np = np.array(prediction)
    ground_truth_np = np.array(ground_truth)

    return roc_auc_score(ground_truth, prediction)

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
def predictWithModels(train_X, train_Y, test_X, predictionFileName):

    # models (classifiers)
    model0 = getLGBMModel(0.15) # lightGBM classifier
    model1 = getLGBMModel(0.1)  # lightGBM classifier
    model2 = getLGBMModel(0.05) # lightGBM classifier
    model3 = getXGBoostModel()  # XGBoost model

    # array of models
    modelNames = ['LGBM0', 'LGBM1', 'LGBM2', 'XGB']
    models = [model0, model1, model2, model3]
    predictions = []

    # predict using these models
    for i in range(len(models)):
        prediction = predict(models[i], train_X, train_Y, test_X, None)
        print('\nprediction using model ' + str(i) + ' :\n' + str(prediction[:5]) + '\n')
        
        # save prediction for each model
        prediction.to_csv(predictionFileName + '_model_' + str(i) + '.csv')
        predictions.append(prediction)

    return predictions

# get merged predictions if merged_predictions_model_X.csv already exists
def getMergedPredictions(numModel):

    merged_predictions = []

    for i in range(numModel):
        merged_predictions.append(pd.read_csv('merged_predictions_model_' + str(i) + '.csv', index_col=0))

    return merged_predictions

# run with normalize / log2 option for INPUT data (True/False each)

# noramlize : noramlize X using average and stddev
# log2      : X -> log2(X+1) if X >= 0
#                  0         if X <  0
def run(train_X, train_Y, test_X, normalize, log2, final, fileID):

    train_rows = 182080
    numModel = 4
    numClass = 1
    
    # apply X -> log2(X+1) to train_X and test_X
    if log2 == True:
        train_X = applyLog(train_X)
        test_X = applyLog(test_X)

    # noramlize train_X and test_X
    if normalize == True:
        train_X = applyNormalization(train_X)
        test_X = applyNormalization(test_X)

    # VALIDATION
    # split training data into train_train and train_valid data using K-fold
    # training using train_train data
    # valid    using train_valid data
    if final == False:

        k = 10
        unit_rows = int(train_rows / k) # rows in a k-folded unit
        score = []

        # the whole merged predictions of each k-folded rows 0~18207, 18208~36415, ..., 163872~182079, for each model
        merged_predictions = np.zeros((numModel, train_rows, numClass))

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
            valid_predictions = predictWithModels(train_train_X, train_train_Y, train_valid_X, 'val' + str(fileID) + '_' + str(i))

            # the whole merged prediction for each model
            for j in range(numModel):
                merged_predictions[j][unit_rows*i : unit_rows*(i+1)] += valid_predictions[j]

        # save merged predictions as *.csv file
        for i in range(numModel):
            pd.DataFrame(merged_predictions[i]).to_csv('merged_predictions_model_' + str(i) + '.csv')
            
        return merged_predictions

    # FINAL PREDICTION
    else: # final == True
        return predictWithModels(train_X, train_Y, test_X, 'fin' + str(fileID))

# compute metric score between merged prediction with weights
def computeMetric(merged_predictions, weights, train_Y):

    # compute merged prediction with the sum of (merged_prediction * weight)
    # shape of both merged_prediction and train_Y : (train_rows, numClass)
    for i in range(len(weights)):
        if i == 0:
            merged_prediction = weights[0] * merged_predictions[0]
        else:
            merged_prediction += weights[i] * merged_predictions[i]
    
    return computeROCAUC(merged_prediction, train_Y)

if __name__ == '__main__':

    # read data
    train_X = pd.read_csv('train_input.csv', index_col=0)
    train_Y = pd.read_csv('train_output.csv', index_col=0)
    test_X = pd.read_csv('test_input.csv', index_col=0)

    # run for at most 200 rounds
    rounds = 200

    # initial weights for each model : [LGBM0, LGBM1, LGBM2, XGB]
    w = [0.25, 0.25, 0.25, 0.25]

    # weight change rate
    wcr = 0.01

    # get merged predictions first
    try:
        merged_predictions = getMergedPredictions(len(w))
    except:
        merged_predictions = run(train_X, train_Y, test_X, False, False, False, 0)

    print(merged_predictions)

    # log
    log = ''

    for i in range(rounds):
        score = computeMetric(merged_predictions, w, train_Y)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' score=' + str(round(score, 8)) + '\n')

        # explore neighboring cases
        w_neighbor0 = [min(w[0] + 3*wcr, 1.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0)]
        w_neighbor1 = [max(w[0] - wcr, 0.0), min(w[1] + 3*wcr, 1.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0)]
        w_neighbor2 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), min(w[2] + 3*wcr, 1.0), max(w[3] - wcr, 0.0)]
        w_neighbor3 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), min(w[3] + 3*wcr, 1.0)]
        
        score_neighbor0 = computeMetric(merged_predictions, w_neighbor0, train_Y)
        score_neighbor1 = computeMetric(merged_predictions, w_neighbor1, train_Y)
        score_neighbor2 = computeMetric(merged_predictions, w_neighbor2, train_Y)
        score_neighbor3 = computeMetric(merged_predictions, w_neighbor3, train_Y)

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using computed scores
        log += ('neighbor0=' + str(np.round_(w_neighbor0, 6)) + ' score=' + str(round(score_neighbor0, 8)) + '\n')
        log += ('neighbor1=' + str(np.round_(w_neighbor1, 6)) + ' score=' + str(round(score_neighbor1, 8)) + '\n')
        log += ('neighbor2=' + str(np.round_(w_neighbor2, 6)) + ' score=' + str(round(score_neighbor2, 8)) + '\n')
        log += ('neighbor3=' + str(np.round_(w_neighbor3, 6)) + ' score=' + str(round(score_neighbor3, 8)) + '\n')
        
        # move to the neighbor with maximum score
        # stop if no neighbor with score greater than 'score'
        scores = np.array([score_neighbor0, score_neighbor1, score_neighbor2, score_neighbor3])
        moveTo = scores.argmax()

        if scores[moveTo] > score:
            for i in range(len(w)):
                if i == moveTo:
                    w[i] = min(w[i] + 3*wcr, 1.0)
                else:
                    w[i] = max(w[i] - wcr, 0.0)
                    
            log += 'move to neighbor' + str(moveTo) + '\n'

        else:
            break

        # modify weight to make the sum 1.0
        sumWeights = sum(w)

        if sumWeights != 1.0:
            for i in range(len(w)):
                w[i] /= sumWeights

            log += 'modify weight to make the sum 1.0\nmodified weight: ' + str(w) + '\n'

    # save log for each round at finish
    f = open('log.txt', 'w')
    f.write(log)
    f.close()

    # final prediction
    final_predictions = run(train_X, train_Y, test_X, False, False, True, 0)

    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * final_predictions[0]
        else:
            final_prediction += w[i] * final_predictions[i]

    # save the final prediction
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.index = range(78035)
    final_prediction.columns = ['final_prediction']
    final_prediction.to_csv('final_prediction.csv')

    # save log for final prediction
    log += ('final prediction with ' + str(w))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
