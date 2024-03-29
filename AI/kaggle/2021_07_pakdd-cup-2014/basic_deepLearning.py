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
    model1 = getBasicNNModel(96)
    model2 = getBasicNNModel(128)
    model3 = getBasicNNModel(192)

    models = [model0, model1, model2, model3]
    isModelNN = [True, True, True, True]
    modelNames = ['NN0', 'NN1', 'NN2', 'NN3']
    numModel = len(models)

    model_info = [models, isModelNN, modelNames]

    # compute or get merged predictions first
    merged_predictions = ph.computeMergedPredictions(train_X, train_Y, test_X, w, train_rows, model_info)

    # find best weights using hill-climbing
    log = ph.findBestWeights(train_Y, merged_predictions, w, wcr, rounds, numModel)

    # final prediction
    final_prediction = ph.getFinalPrediction(w, train_X, train_Y, test_X, train_rows, model_info)

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
