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

# models : CatBoost and lightGBM
def getCatBoostModel(iterations, depth):

    # catboost classifier
    # ref: https://www.kaggle.com/tharunreddy/tps-june-catboost-tutorial
    model = CatBoostClassifier(iterations=iterations,
                               learning_rate=0.028,
                               depth=depth, # 10
                               loss_function='MultiClass',
                               min_data_in_leaf=11,
                               reg_lambda=53.3,
                               random_strength=32,
                               verbose=10)

    return model

def getLGBMModel(learning_rate):

    # LGBM classifier
    # ref: https://www.kaggle.com/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding
    params = {
        'n_estimators': 10000,
        'learning_rate': learning_rate, # 0.2423
        'num_leaves': 2260,
        'max_depth': 9,
        'min_data_in_leaf': 8600,
        'lambda_l1': 70,
        'lambda_l2': 35,
        'min_gain_to_split': 0.11775,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
        'feature_fraction': 0.6,
        'random_state': 2021,
        'verbose': 10
    }
    model = LGBMClassifier(**params)

    return model

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName):
    print(' ==== predict using model : ' + str(model) + ' ====')
                            
    model.fit(train_X, train_Y)
    
    prediction = model.predict_proba(test_X)
    #prediction = np.clip(prediction, 0.08, 0.95)
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
    model0 = getCatBoostModel(500, 4)  # catboost classifier
    model1 = getCatBoostModel(1000, 4) # catboost classifier
    model2 = getCatBoostModel(2000, 4) # catboost classifier
    model3 = getCatBoostModel(500, 6)  # catboost classifier
    model4 = getCatBoostModel(1000, 6) # catboost classifier
    model5 = getCatBoostModel(2000, 6) # catboost classifier
    #model3 = getLGBMModel(0.12)     # lightGBM classifier
    #model4 = getLGBMModel(0.24)     # lightGBM classifier
    #model5 = getLGBMModel(0.36)     # lightGBM classifier

    # array of models
    modelNames = ['catboost0', 'catboost1', 'catboost2', 'catboost3', 'catboost4', 'catboost5']
    models = [model0, model1, model2, model3, model4, model5]
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

    train_rows = 200000
    numClass = 9
    numModel = 6
    k = 10 # for k-fold
    
    # extract training and test data
    train_X = train_df.loc[:, 'feature_0':'feature_74']
    train_Y = train_df.loc[:, 'target':]
    test_X = test_df.loc[:, 'feature_0':'feature_74']
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

        unit_rows = int(train_rows / k) # rows in a k-folded unit
        error = []

        # the whole merged predictions of each k-folded rows 0 ~ L-1, L ~ 2L-1, ..., (k-1)L ~ kL-1, for each model
        # where L is the length of each unit
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

            print('valid_predictions')
            print(np.shape(valid_predictions))

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
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_Y = train_df.loc[:, 'target']

    # run for at most 200 rounds
    rounds = 200
    models = 6
    wcr = 1/240 # weight change rate
    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3, 'Class_5':4, 'Class_6':5, 'Class_7':6, 'Class_8':7, 'Class_9':8}

    # get merged predictions first
    try:
        merged_predictions = getMergedPredictions(len(w))
    except:
        merged_predictions = run(train_df, test_df, dic,
                                 normalize=False, log2=True, final=False, fileID=0)

    print(merged_predictions)

    # initial weights for each model : [CatBoost0, CatBoost1, CatBoost2, LGBM0, LGBM1, LGBM2, LDA]
    w = []
    for i in range(models): w.append(1/models)

    # log
    log = ''

    for i in range(rounds):
        error = computeError(merged_predictions, w, train_Y)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' error=' + str(round(error, 8)) + '\n')

        # explore neighboring cases
        w_neighbors = []     
        for j in range(models):
            w_neighbor = []
            for k in range(models):
                if j == k:
                    w_neighbor.append(min(w[k] + (models-1)*wcr, 1.0))
                else:
                    w_neighbor.append(max(w[0] - wcr, 0.0))
            w_neighbors.append(w_neighbor)

        error_neighbors = []
        for i in range(models):
            error_neighbors.append(computeError(merged_predictions, w_neighbors[i], train_Y))

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using meanError
        for i in range(models):
            log += ('neighbor' + str(i) + '=' + str(np.round_(w_neighbors[i], 6)) + ' error=' + str(round(error_neighbors[i], 8)) + '\n')

        # move to the neighbor with minimum loss
        # stop if no neighbor with loss less than 'error'
        moveTo = error_neighbors.argmin()

        if error_neighbors[moveTo] < error:
            for i in range(len(w)):
                if i == moveTo:
                    w[i] = min(w[i] + (models-1)*wcr, 1.0)
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
    final_predictions = run(train_df, test_df, dic,
                            normalize=False, log2=True, final=True, fileID=0)

    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * final_predictions[0]
        else:
            final_prediction += w[i] * final_predictions[i]

    # change row number and column as 'sample_submission.csv'
    # you need to write 'id' to the top-left cell
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.index = range(200000, 300000)
    final_prediction.columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    final_prediction.to_csv('final_prediction.csv')

    # save log for final prediction
    log += ('final prediction with ' + str(w))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
