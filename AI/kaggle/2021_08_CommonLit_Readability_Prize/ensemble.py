# use template of https://github.com/WannaBeSuperteur/2020/blob/master/AI/kaggle/2021_06_Tabular_Playground_May_2021/Code/KimHongSik_tps-may-2021-eda-catboost-lightautoml.py

# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import numpy as np 
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential

import predict_helper as ph

from sklearn.metrics import mean_squared_error

# compute error between merged prediction with weights
def computeError(validData_, validOriginal, w, loss):

    # compute merged prediction with the sum of (merged_prediction * weight)
    for i in range(len(w)):
        if i == 0:
            merged_prediction = w[0] * validData_[0]
        else:
            merged_prediction += w[i] * validData_[i]

    if loss == 'RMSE': return pow(mean_squared_error(merged_prediction, validOriginal), 0.5)

# find best weights using hill-climbing
# models : the number of models
def findBestWeights(validData_, validOriginal, w, train_rows, models, loss):

    # log
    log = ''

    for i in range(rounds):
        error = computeError(validData_, validOriginal, w, loss)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' error=' + str(round(error, 8)) + '\n')

        # explore neighboring cases
        w_neighbors = []     
        for j in range(models):
            w_neighbor = []
            for k in range(models):
                if j == k:
                    w_neighbor.append(min(w[k] + (models-1)*wcr, 1.0))
                else:
                    w_neighbor.append(max(w[k] - wcr, 0.0))
            w_neighbors.append(w_neighbor)

        error_neighbors = []
        for i in range(models):
            error_neighbors.append(computeError(validData_, validOriginal, w_neighbors[i], loss))

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using meanError
        for i in range(models):
            log += ('neighbor' + str(i) + '=' + str(np.round_(w_neighbors[i], 6)) + ' error=' + str(round(error_neighbors[i], 8)) + '\n')

        # move to the neighbor with minimum loss
        # stop if no neighbor with loss less than 'error'
        moveTo = np.array(error_neighbors).argmin()

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

    return log

# final prediction
def getFinalPrediction(testData_, w, models):
    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * testData_[0]
        else:
            final_prediction += w[i] * testData_[i]

    return final_prediction

if __name__ == '__main__':

    train_rows = 2834
    models = 40
    rounds = 2000 # run for at most 2000 rounds
    wcr = 0.001 # weight change rate
    loss = 'RMSE'

    # read original validation data
    validOriginal = pd.read_csv('valid_original.csv', index_col=0)

    # read validation data
    validData = ['valid_roberta_0.csv', 'valid_roberta_1.csv', 'valid_roberta_2.csv', 'valid_roberta_3.csv',
                 'valid_roberta_4.csv', 'valid_roberta_5.csv', 'valid_roberta_6.csv', 'valid_roberta_7.csv',
                 'valid_roberta_8.csv', 'valid_roberta_9.csv', 'valid_roberta_10.csv', 'valid_roberta_11.csv',
                 'valid_roberta_12.csv', 'valid_roberta_13.csv', 'valid_roberta_14.csv', 'valid_roberta_15.csv',
                 'valid_roberta_16.csv', 'valid_roberta_17.csv', 'valid_roberta_18.csv', 'valid_roberta_19.csv',
                 
                 'valid_distilbert_0.csv', 'valid_distilbert_1.csv', 'valid_distilbert_2.csv', 'valid_distilbert_3.csv',
                 'valid_distilbert_4.csv', 'valid_distilbert_5.csv', 'valid_distilbert_6.csv', 'valid_distilbert_7.csv',
                 'valid_distilbert_8.csv', 'valid_distilbert_9.csv', 'valid_distilbert_10.csv', 'valid_distilbert_11.csv',
                 'valid_distilbert_12.csv', 'valid_distilbert_13.csv', 'valid_distilbert_14.csv', 'valid_distilbert_15.csv',
                 'valid_distilbert_16.csv', 'valid_distilbert_17.csv', 'valid_distilbert_18.csv', 'valid_distilbert_19.csv']

    validData_ = []
    for i in range(models):
        validData_.append(pd.read_csv(validData[i], index_col=0))

    # read test data
    testData = ['test_roberta_0.csv', 'test_roberta_1.csv', 'test_roberta_2.csv', 'test_roberta_3.csv',
                'test_roberta_4.csv', 'test_roberta_5.csv', 'test_roberta_6.csv', 'test_roberta_7.csv',
                'test_roberta_8.csv', 'test_roberta_9.csv', 'test_roberta_10.csv', 'test_roberta_11.csv',
                'test_roberta_12.csv', 'test_roberta_13.csv', 'test_roberta_14.csv', 'test_roberta_15.csv',
                'test_roberta_16.csv', 'test_roberta_17.csv', 'test_roberta_18.csv', 'test_roberta_19.csv',
                
                'test_distilbert_0.csv', 'test_distilbert_1.csv', 'test_distilbert_2.csv', 'test_distilbert_3.csv',
                'test_distilbert_4.csv', 'test_distilbert_5.csv', 'test_distilbert_6.csv', 'test_distilbert_7.csv',
                'test_distilbert_8.csv', 'test_distilbert_9.csv', 'test_distilbert_10.csv', 'test_distilbert_11.csv',
                'test_distilbert_12.csv', 'test_distilbert_13.csv', 'test_distilbert_14.csv', 'test_distilbert_15.csv',
                'test_distilbert_16.csv', 'test_distilbert_17.csv', 'test_distilbert_18.csv', 'test_distilbert_19.csv']

    testData_ = []
    for i in range(models):
        testData_.append(pd.read_csv(testData[i], index_col=0))

    # read model name
    modelName = ['valid_roberta_0', 'valid_roberta_1', 'valid_roberta_2', 'valid_roberta_3',
                 'valid_roberta_4', 'valid_roberta_5', 'valid_roberta_6', 'valid_roberta_7',
                 'valid_roberta_8', 'valid_roberta_9', 'valid_roberta_10', 'valid_roberta_11',
                 'valid_roberta_12', 'valid_roberta_13', 'valid_roberta_14', 'valid_roberta_15',
                 'valid_roberta_16', 'valid_roberta_17', 'valid_roberta_18', 'valid_roberta_19',
                 
                 'valid_distilbert_0', 'valid_distilbert_1', 'valid_distilbert_2', 'valid_distilbert_3',
                 'valid_distilbert_4', 'valid_distilbert_5', 'valid_distilbert_6', 'valid_distilbert_7',
                 'valid_distilbert_8', 'valid_distilbert_9', 'valid_distilbert_10', 'valid_distilbert_11',
                 'valid_distilbert_12', 'valid_distilbert_13', 'valid_distilbert_14', 'valid_distilbert_15',
                 'valid_distilbert_16', 'valid_distilbert_17', 'valid_distilbert_18', 'valid_distilbert_19']

    models_ = []
    for i in range(models):
        models_.append(tf.keras.models.load_model(modelName[i]))

    # initial weights for each model
    w = [1/models for i in range(models)]

    # find best weights using hill-climbing
    log = findBestWeights(validData_, validOriginal, w, train_rows, models, loss)

    # final prediction
    final_prediction = getFinalPrediction(testData_, w, models)

    # save the final prediction
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.index = range(len(final_prediction))
    final_prediction.columns = ['final_prediction_by_ensemble']
    final_prediction.to_csv('final_prediction_by_ensemble.csv')

    # save log for final prediction
    log += ('final prediction with ' + str(w))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
