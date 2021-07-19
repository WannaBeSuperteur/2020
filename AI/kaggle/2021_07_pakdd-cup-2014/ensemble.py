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

import lightgbm
import xgboost
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# XGBoost model
# ref : https://www.kaggle.com/safavieh/ultimate-feature-engineering-xgb-lgb-nn
def getXGBoostModel(max_depth, min_child_weight):

    params = {
        'eta': 0.05,
        'max_depth': max_depth,
        'subsample': 0.85,
        'colsample_bytree': 0.25,
        'min_child_weight': min_child_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1,
    }

    model = XGBRegressor(**params)
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
    model0 = getXGBoostModel(4, 3)
    model1 = getXGBoostModel(4, 4)
    model2 = getXGBoostModel(5, 3)
    model3 = getXGBoostModel(5, 4)

    models = [model0, model1, model2, model3]
    isModelNN = [False, False, False, False]
    modelNames = ['LGBM0', 'LGBM1', 'LGBM2', 'XGBOOST']
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
