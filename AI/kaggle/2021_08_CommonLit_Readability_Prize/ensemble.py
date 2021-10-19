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
def findBestWeights(validData_, validOriginal, w, wcr, train_rows, models, loss, rounds):

    # log
    log = ''

    for i in range(rounds):
        if i % 5 == 0: print('round', i)
        
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
    models = 9
    rounds = 500 # run for at most 500 rounds
    wcr = 1 / 180 # weight change rate
    loss = 'RMSE'

    # read original validation data
    validOriginal = pd.read_csv('valid_original.csv', index_col=0)

    # read validation data
    validData = ['valid_LSTM0_0.csv', 'valid_LSTM0_1.csv', 'valid_LSTM0_2.csv',
                 'valid_LSTM0_3.csv', 'valid_LSTM0_4.csv', 'valid_LSTM0_5.csv',
                 'valid_LSTM0_6.csv', 'valid_LSTM0_7.csv', 'valid_LSTM0_8.csv']
                 #'valid_roberta_0.csv', 'valid_roberta_1.csv', 'valid_roberta_2.csv', 'valid_roberta_3.csv', 'valid_roberta_4.csv',
                 #'valid_distilbert_0.csv', 'valid_distilbert_1.csv', 'valid_distilbert_2.csv', 'valid_distilbert_3.csv', 'valid_distilbert_4.csv']

    validData_ = []
    for i in range(models):
        validData_.append(pd.read_csv(validData[i], index_col=0))

    # read test data
    testData = ['test_LSTM0_0.csv', 'test_LSTM0_1.csv', 'test_LSTM0_2.csv',
                'test_LSTM0_3.csv', 'test_LSTM0_4.csv', 'test_LSTM0_5.csv',
                'test_LSTM0_6.csv', 'test_LSTM0_7.csv', 'test_LSTM0_8.csv']
                #'test_roberta_0.csv', 'test_roberta_1.csv', 'test_roberta_2.csv', 'test_roberta_3.csv', 'test_roberta_4.csv',
                #'test_distilbert_0.csv', 'test_distilbert_1.csv', 'test_distilbert_2.csv', 'test_distilbert_3.csv', 'test_distilbert_4.csv']

    testData_ = []
    for i in range(models):
        testData_.append(pd.read_csv(testData[i], index_col=0))

    # denormalize
    # Z = (X-m)/s -> X = Z*s+m
    normalizeInfo = pd.read_csv('normalization_info.csv', index_col=0)
    average       = normalizeInfo['avg'][0]
    stddev        = normalizeInfo['stddev'][0]

    print('\nnormalization info:')
    print(normalizeInfo, average, stddev)

    validOriginal = np.array(validOriginal) * stddev + average
    validData_    = np.array(validData_)    * stddev + average
    testData_     = np.array(testData_)     * stddev + average

    # read model name
    modelName = ['main_LSTM0_0', 'main_LSTM0_1', 'main_LSTM0_2',
                 'main_LSTM0_3', 'main_LSTM0_4', 'main_LSTM0_5',
                 'main_LSTM0_6', 'main_LSTM0_7', 'main_LSTM0_8']
                 #'valid_roberta_0', 'valid_roberta_1', 'valid_roberta_2', 'valid_roberta_3', 'valid_roberta_4',
                 #'valid_distilbert_0', 'valid_distilbert_1', 'valid_distilbert_2', 'valid_distilbert_3', 'valid_distilbert_4']

    models_ = []
    for i in range(models):
        models_.append(tf.keras.models.load_model(modelName[i]))

    # initial weights for each model
    w = [1/models for i in range(models)]

    # find best weights using hill-climbing
    log = findBestWeights(validData_, validOriginal, w, wcr, train_rows, models, loss, rounds)

    # final prediction (already denormalized)
    final_prediction = np.array(getFinalPrediction(testData_, w, models))

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
