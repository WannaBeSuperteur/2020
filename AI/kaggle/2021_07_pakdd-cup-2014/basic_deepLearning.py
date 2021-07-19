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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

import predict_helper as ph

# basic NN model
# ref : https://github.com/WannaBeSuperteur/2020/blob/master/WPCN_UAV/2020_10_Multi_UAV_Enabled/deepQ.py
def getBasicNNModel(units):

    with tf.device('/gpu:0'):

        # L2 regularization
        L2 = tf.keras.regularizers.l2(0.001)

        # define the model
        model = Sequential()
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=L2))
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=L2))
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=L2))
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=L2))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        return model

if __name__ == '__main__':

    train_rows = 54388

    # read data
    train_X = pd.read_csv('train_X.csv', index_col=0)
    train_Y = pd.read_csv('train_Y.csv', index_col=0)
    test_X = pd.read_csv('test_X.csv', index_col=0)

    # run for at most 200 rounds
    rounds = 200

    # initial weights for each model : [NN0, NN1, NN2, NN3]
    w = [0.25, 0.25, 0.25, 0.25]

    # weight change rate
    wcr = 0.01

    # model info
    model0 = getBasicNNModel(64)
    model1 = getBasicNNModel(128)
    model2 = getBasicNNModel(256)
    model3 = getBasicNNModel(512)

    models = [model0, model1, model2, model3]
    isModelNN = [True, True, True, True]
    modelNames = ['NN0', 'NN1', 'NN2', 'NN3']

    model_info = [models, isModelNN, modelNames]

    # get merged predictions first
    try:
        merged_predictions = ph.getMergedPredictions(len(w))
    except:
        merged_predictions = ph.run(train_X, train_Y, test_X, False, 0, train_rows, model_info)

    print(merged_predictions)

    # log
    log = ''

    for i in range(rounds):
        error = ph.computeMetric(merged_predictions, w, train_Y)
        log += ('\n[ round ' + str(i) + ' ]\nweights=' + str(np.round_(w, 6)) + ' error=' + str(round(error, 8)) + '\n')

        # explore neighboring cases
        w_neighbor0 = [min(w[0] + 3*wcr, 1.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0)]
        w_neighbor1 = [max(w[0] - wcr, 0.0), min(w[1] + 3*wcr, 1.0), max(w[2] - wcr, 0.0), max(w[3] - wcr, 0.0)]
        w_neighbor2 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), min(w[2] + 3*wcr, 1.0), max(w[3] - wcr, 0.0)]
        w_neighbor3 = [max(w[0] - wcr, 0.0), max(w[1] - wcr, 0.0), max(w[2] - wcr, 0.0), min(w[3] + 3*wcr, 1.0)]
        
        error_neighbor0 = ph.computeMetric(merged_predictions, w_neighbor0, train_Y)
        error_neighbor1 = ph.computeMetric(merged_predictions, w_neighbor1, train_Y)
        error_neighbor2 = ph.computeMetric(merged_predictions, w_neighbor2, train_Y)
        error_neighbor3 = ph.computeMetric(merged_predictions, w_neighbor3, train_Y)

        # save log for each round
        f = open('log.txt', 'w')
        f.write(log)
        f.close()

        # modify weights using computed losses
        log += ('neighbor0=' + str(np.round_(w_neighbor0, 6)) + ' error=' + str(round(error_neighbor0, 8)) + '\n')
        log += ('neighbor1=' + str(np.round_(w_neighbor1, 6)) + ' error=' + str(round(error_neighbor1, 8)) + '\n')
        log += ('neighbor2=' + str(np.round_(w_neighbor2, 6)) + ' error=' + str(round(error_neighbor2, 8)) + '\n')
        log += ('neighbor3=' + str(np.round_(w_neighbor3, 6)) + ' error=' + str(round(error_neighbor3, 8)) + '\n')
        
        # move to the neighbor with minimum loss
        # stop if no neighbor with loss less than 'error'
        errors = np.array([error_neighbor0, error_neighbor1, error_neighbor2, error_neighbor3])
        moveTo = errors.argmin()

        if errors[moveTo] < error:
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
        log += ph.modifySumTo1(w)
        
    # save log for each round at finish
    f = open('log.txt', 'w')
    f.write(log)
    f.close()

    # final prediction
    final_predictions = ph.run(train_X, train_Y, test_X, True, 0, train_rows, model_info)

    for i in range(len(w)):
        if i == 0:
            final_prediction = w[0] * final_predictions[0]
        else:
            final_prediction += w[i] * final_predictions[i]

    # save the final prediction
    final_prediction = pd.DataFrame(final_prediction)
    final_prediction.index = range(4256)
    final_prediction.columns = ['final_prediction']
    final_prediction.to_csv('final_prediction.csv')

    # save log for final prediction
    log += ('final prediction with ' + str(w))
    
    f = open('log.txt', 'w')
    f.write(log)
    f.close()
