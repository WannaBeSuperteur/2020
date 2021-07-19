# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

# predict and save the prediction
def predict(model, train_X, train_Y, test_X, predictionFileName, isModelNN):
    print(' ==== predict using model : ' + str(model) + ' ====')

    train_X_array = np.array(train_X)
    train_Y_array = np.array(train_Y)
    test_X_array = np.array(test_X)

    if isModelNN == True:
        with tf.device('/gpu:0'):
            model.fit(train_X_array, train_Y_array, batch_size=32, epochs=15)
    else:
        model.fit(train_X_array, train_Y_array)

    prediction = model.predict(test_X_array)
    prediction = pd.DataFrame(prediction).iloc[:, 0:1]

    # save prediction
    if predictionFileName != None:
        prediction.to_csv(predictionFileName + '.csv')
        
    return prediction

# compute and return MAE loss
# ground_truth : label      , shape=(rows, 1)
# prediction   : probability, shape=(rows, numClass)
def computeMAE(prediction, ground_truth):

    prediction_np = np.array(prediction)
    ground_truth_np = np.array(ground_truth)

    return mean_absolute_error(ground_truth, prediction)

# predict using multiple models
def predictWithModels(train_X, train_Y, test_X, predictionFileName, model_info):

    # models (classifiers)
    models = model_info[0]
    isModelNN = model_info[1]
    modelNames = model_info[2]
    
    predictions = []

    # predict using these models
    for i in range(len(models)):
        prediction = predict(models[i], train_X, train_Y, test_X, None, isModelNN[i])
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

# noramlize  : noramlize X using average and stddev
# log2       : X -> log2(X+1) if X >= 0
#                   0         if X <  0
# model_info : (for example,)
# model_info = [models, isModelNN, modelNames]
#              where models     = [model0, model1, model2, model3],
#                    isModelNN  = [True, True, True, True],
#                    modelNames = ['NN0', 'NN1', 'NN2', 'NN3']
def run(train_X, train_Y, test_X, final, fileID, train_rows, model_info):

    numModel = len(model_info[0])

    # VALIDATION
    # split training data into train_train and train_valid data using K-fold
    # training using train_train data
    # valid    using train_valid data
    if final == False:

        k = 10
        unit_rows = int(train_rows / k) # rows in a k-folded unit

        # the whole merged predictions of each k-folded rows 0%~10%, 10%~20%, ..., 90%~100% (of entire rows), for each model
        merged_predictions = np.zeros((numModel, train_rows, 1))

        for i in range(k):
        
            train_train_X = pd.concat([train_X.loc[:unit_rows*i - 1, :], train_X.loc[unit_rows*(i+1):, :]])
            train_train_Y = pd.concat([train_Y.loc[:unit_rows*i - 1, :], train_Y.loc[unit_rows*(i+1):, :]])
            train_valid_X = train_X.loc[unit_rows*i : unit_rows*(i+1) - 1, :]
            train_valid_Y = train_Y.loc[unit_rows*i : unit_rows*(i+1) - 1, :]

            print(' ======== normalize: FALSE log2: FALSE ========')
            print(train_train_X)
            print(train_train_Y)
            print(train_valid_X)
            print(train_valid_Y)
            print(' ========================================')

            # validation
            valid_predictions = predictWithModels(train_train_X, train_train_Y, train_valid_X,
                                                  'val' + str(fileID) + '_' + str(i), model_info)

            # the whole merged prediction for each model
            for j in range(numModel):
                merged_predictions[j][unit_rows*i : unit_rows*(i+1)] += valid_predictions[j]

        # save merged predictions as *.csv file
        for i in range(numModel):
            pd.DataFrame(merged_predictions[i]).to_csv('merged_predictions_model_' + str(i) + '.csv')
            
        return merged_predictions

    # FINAL PREDICTION
    else: # final == True
        return predictWithModels(train_X, train_Y, test_X,
                                 'fin' + str(fileID), model_info)

# compute metric score (or loss) between merged prediction with weights
def computeMetric(merged_predictions, weights, train_Y):

    # compute merged prediction with the sum of (merged_prediction * weight)
    # shape of both merged_prediction and train_Y : (train_rows, numClass)
    for i in range(len(weights)):
        if i == 0:
            merged_prediction = weights[0] * merged_predictions[0]
        else:
            merged_prediction += weights[i] * merged_predictions[i]
    
    return computeMAE(merged_prediction, train_Y)

# make the sum 1.0 and return the message
def modifySumTo1(w):
    sumWeights = sum(w)

    if sumWeights != 1.0:
        for i in range(len(w)):
            w[i] /= sumWeights

        return 'modify weight to make the sum 1.0\nmodified weight: ' + str(w) + '\n'
    else:
        return ''

# compute merged predictions
def computeMergedPredictions(train_X, train_Y, test_X, w, train_rows, model_info):
    try:
        merged_predictions = getMergedPredictions(len(w))
    except:
        merged_predictions = run(train_X, train_Y, test_X, False, 0, train_rows, model_info)

    print(merged_predictions)
    return merged_predictions

# find best weights (w) using hill-climbing and return the log
def findBestWeights(train_Y, merged_predictions, w, wcr, rounds, numModel):
    
    # log
    log = ''

    for i in range(rounds):
        error = computeMetric(merged_predictions, w, train_Y)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' error=' + str(round(error, 8)) + '\n')

        # explore neighboring cases
        w_neighbors = []
        error_neighbors = []
        
        for j in range(numModel):
            w_neighbor = []

            for k in range(numModel):
                if j == k:
                    w_neighbor.append(min(w[k] + (numModel - 1)*wcr, 1.0))
                else:
                    w_neighbor.append(max(w[k] - wcr, 0.0))
            
            w_neighbors.append(w_neighbor)
            error_neighbors.append(computeMetric(merged_predictions, w_neighbor, train_Y))

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using computed losses
        for j in range(numModel):
            log += ('neighbor' + str(j) + '=' + str(np.round_(w_neighbors[j], 6)) + ' error=' + str(round(error_neighbors[j], 8)) + '\n')
        
        # move to the neighbor with minimum loss
        # stop if no neighbor with loss less than 'error'
        errors = np.array(error_neighbors)
        moveTo = errors.argmin()

        if errors[moveTo] < error:
            for i in range(len(w)):
                if i == moveTo:
                    w[i] = min(w[i] + (numModel - 1)*wcr, 1.0)
                else:
                    w[i] = max(w[i] - wcr, 0.0)
                    
            log += 'move to neighbor' + str(moveTo) + '\n'

        else:
            break

        # modify weight to make the sum 1.0
        sumWeights = sum(w)
        log += modifySumTo1(w)
        
    # save log for each round at finish
    f = open('log.txt', 'w')
    f.write(log)
    f.close()

    return log

# get final prediction with weights (w)
def getFinalPrediction(w, train_X, train_Y, test_X, train_rows, model_info):
    final_predictions = run(train_X, train_Y, test_X, True, 0, train_rows, model_info)

    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * final_predictions[0]
        else:
            final_prediction += w[i] * final_predictions[i]

    return final_prediction
