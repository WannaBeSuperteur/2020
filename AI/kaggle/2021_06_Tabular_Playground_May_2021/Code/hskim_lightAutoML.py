# using
# https://www.kaggle.com/danieljhk/top-2-lightautoml-stacking
# https://www.kaggle.com/azzamradman/top-2-lightautoml-stacking

# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

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

# run with normalize / log2 option for INPUT data (True/False each)

# noramlize : noramlize X using average and stddev
# log2      : X -> log2(X+1) if X >= 0
#                  0         if X <  0
def run(train_df, test_df, dic, normalize, log2):

    train_rows = 100000
    
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
    # CHANGED TabularAutoML to TabularUtilizedAutoML for timeout utilization
    N_THREADS = 4
    N_FOLDS = 10
    RANDOM_STATE = 2021
    TEST_SIZE = 0.1
    TIMEOUT = 1800

    task = Task('multiclass', metric='crossentropy')
    roles = {'target': 'target'}
    
    automl = TabularUtilizedAutoML(task=task,
                                   timeout=TIMEOUT,
                                   cpu_limit=4,
                                   general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
                                   reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE})
    
    automl_prediction = automl.fit_predict(train_df, roles=roles)

    test_Y = automl.predict(test_X)
    test_Y.to_csv('automl_final_prediction.csv')

if __name__ == '__main__':

    # read data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')
    train_Y = train_df.loc[:, 'target']

    dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3}

    # final prediction using AutoML
    final_predictions = run(train_df, test_df, dic, True, True)

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
