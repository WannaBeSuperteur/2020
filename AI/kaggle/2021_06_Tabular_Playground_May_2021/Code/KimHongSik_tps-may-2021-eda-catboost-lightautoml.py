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
import xgboost
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# team members' Python codes
import Daniel_BaselineModels as Daniel_Baseline

# models
# XGBoost            : An Jun Hang's idea
# LR, LDA, ..., LGBM : Daniel's idea
# CatBoost           : Hong Sik Kim's idea
def getCatBoostModel(iterations):

    # catboost classifier
    model = CatBoostClassifier(iterations=iterations,
                            learning_rate=0.01,
                            depth=4,
                            loss_function='MultiClass',
                            od_wait=1000,
                            od_type='Iter',                           
                            min_data_in_leaf=1,
                            max_ctr_complexity=15,
                            verbose=400)

    return model

# lightGBM (ref: Daniel's Kaggle - https://www.kaggle.com/danieljhk/lgbm-baseline/output?scriptVersionId=67233275)
def getLGBMModel(learning_rate):

    # LGBM classifier
    params = {
        'learning_rate': learning_rate,
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

# Linear Discriminant Analysis ( https://github.com/WannaBeSuperteur/2020/blob/master/AI/kaggle/2021_06_Tabular_Playground_May_2021/Code/Daniel_BaselineModels.py
#                                https://github.com/WannaBeSuperteur/2020/blob/master/AI/kaggle/2021_06_Tabular_Playground_May_2021/Code/Daniel_BaselineModelScores.csv )
def getLDAModel():
    return LinearDiscriminantAnalysis()

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName):
    print(' ==== predict using model : ' + str(model) + ' ====')
                            
    model.fit(train_X, train_Y)
    
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
def predictWithModels(train_X, train_Y, test_X, predictionFileName):

    # models (classifiers)
    model0 = getCatBoostModel(4200)  # catboost classifier
    model1 = getCatBoostModel(8500)  # catboost classifier
    model2 = getCatBoostModel(17000) # catboost classifier
    model3 = getLGBMModel(0.055)     # lightGBM classifier
    model4 = getLGBMModel(0.111)     # lightGBM classifier
    model5 = getLGBMModel(0.223)     # lightGBM classifier
    model6 = getLDAModel()           # Linear Discriminant Analysis

    # array of models
    modelNames = ['catboost0', 'catboost1', 'catboost2', 'lightgbm0', 'lightgbm1', 'lightgbm2', 'lda']
    models = [model0, model1, model2, model3, model4, model5, model6]
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
def run(train_df, test_df, dic, normalize, log2, final, fileID):

    train_rows = 100000
    numClass = 4
    numModel = 7
    
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

    # VALIDATION
    # split training data into train_train and train_valid data using K-fold
    # training using train_train data
    # valid    using train_valid data
    if final == False:

        k = 10
        unit_rows = int(train_rows / k) # rows in a k-folded unit
        error = []

        # the whole merged predictions of each k-folded rows 0~9999, 10000~19999, ..., 90000~99999, for each model
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

# compute error between merged prediction with weights
def computeError(merged_predictions, weights, train_Y):

    # compute merged prediction with the sum of (merged_prediction * weight)
    # shape of both merged_prediction and train_Y : (train_rows, numClass)
    for i in range(len(weights)):
        if i == 0:
            merged_prediction = weights[0] * merged_predictions[0]
        else:
            merged_prediction += weights[i] * merged_predictions[i]
    
    return computeMulticlassLoss(merged_prediction, train_Y)

if __name__ == '__main__':

    # read data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')
    train_Y = train_df.loc[:, 'target']

    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3}

    # run for at most 200 rounds
    rounds = 200

    # initial weights for each model : [CatBoost0, CatBoost1, CatBoost2, LGBM0, LGBM1, LGBM2, LDA]
    w = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]

    # weight change rate
    wcr = 1/280

    # get merged predictions first
    try:
        merged_predictions = getMergedPredictions(len(w))
    except:
        merged_predictions = run(train_df, test_df, dic, False, False, False, 0)

    print(merged_predictions)

    # log
    log = ''

    for i in range(rounds):
        error = computeError(merged_predictions, w, train_Y)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' error=' + str(round(error, 8)) + '\n')

        # explore neighboring cases
        w_neighbor0 = [min(w[0] + 6*wcr, 1.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0), max(w[4] - wcr, 0.0), max(w[5] - wcr, 0.0), max(w[6] - wcr, 0.0)]
        w_neighbor1 = [max(w[0] - wcr, 0.0), min(w[1] + 6*wcr, 1.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0), max(w[4] - wcr, 0.0), max(w[5] - wcr, 0.0), max(w[6] - wcr, 0.0)]
        w_neighbor2 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), min(w[2] + 6*wcr, 1.0), max(w[3] - wcr, 0.0), max(w[4] - wcr, 0.0), max(w[5] - wcr, 0.0), max(w[6] - wcr, 0.0)]
        w_neighbor3 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), min(w[3] + 6*wcr, 1.0), max(w[4] - wcr, 0.0), max(w[5] - wcr, 0.0), max(w[6] - wcr, 0.0)]
        w_neighbor4 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0), min(w[4] + 6*wcr, 1.0), max(w[5] - wcr, 0.0), max(w[6] - wcr, 0.0)]
        w_neighbor5 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0), max(w[4] - wcr, 0.0), min(w[5] + 6*wcr, 1.0), max(w[6] - wcr, 0.0)]
        w_neighbor6 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0), max(w[4] - wcr, 0.0), max(w[5] - wcr, 0.0), min(w[6] + 6*wcr, 1.0)]
        
        error_neighbor0 = computeError(merged_predictions, w_neighbor0, train_Y)
        error_neighbor1 = computeError(merged_predictions, w_neighbor1, train_Y)
        error_neighbor2 = computeError(merged_predictions, w_neighbor2, train_Y)
        error_neighbor3 = computeError(merged_predictions, w_neighbor3, train_Y)
        error_neighbor4 = computeError(merged_predictions, w_neighbor4, train_Y)
        error_neighbor5 = computeError(merged_predictions, w_neighbor5, train_Y)
        error_neighbor6 = computeError(merged_predictions, w_neighbor6, train_Y)

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using meanError
        log += ('neighbor0=' + str(np.round_(w_neighbor0, 6)) + ' error=' + str(round(error_neighbor0, 8)) + '\n')
        log += ('neighbor1=' + str(np.round_(w_neighbor1, 6)) + ' error=' + str(round(error_neighbor1, 8)) + '\n')
        log += ('neighbor2=' + str(np.round_(w_neighbor2, 6)) + ' error=' + str(round(error_neighbor2, 8)) + '\n')
        log += ('neighbor3=' + str(np.round_(w_neighbor3, 6)) + ' error=' + str(round(error_neighbor3, 8)) + '\n')
        log += ('neighbor4=' + str(np.round_(w_neighbor4, 6)) + ' error=' + str(round(error_neighbor4, 8)) + '\n')
        log += ('neighbor5=' + str(np.round_(w_neighbor5, 6)) + ' error=' + str(round(error_neighbor5, 8)) + '\n')
        log += ('neighbor6=' + str(np.round_(w_neighbor6, 6)) + ' error=' + str(round(error_neighbor6, 8)) + '\n')

        # move to the neighbor with minimum loss
        # stop if no neighbor with loss less than 'error'
        errors = np.array([error_neighbor0, error_neighbor1, error_neighbor2, error_neighbor3, error_neighbor4, error_neighbor5, error_neighbor6])
        moveTo = errors.argmin()

        if errors[moveTo] < error:
            for i in range(len(w)):
                if i == moveTo:
                    w[i] = min(w[i] + 6*wcr, 1.0)
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
    final_predictions = run(train_df, test_df, dic, False, False, True, 0)

    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * final_predictions[0]
        else:
            final_prediction += w[i] * final_predictions[i]

    # change row number and column as 'sample_submission.csv'
    # you need to write 'id' to the top-left cell
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.index = range(100000, 150000)
    final_prediction.columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4']
    final_prediction.to_csv('final_prediction.csv')

    # save log for final prediction
    log += ('final prediction with ' + str(w))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
