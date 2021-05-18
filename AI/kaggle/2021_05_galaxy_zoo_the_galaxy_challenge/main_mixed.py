import pandas as pd
import numpy as np
import math
import random

import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import deepLearning_main as DL

import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

try:
    from keras import backend as K
except:
    import keras.backend.tensorflow_backend as K

import used_model

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')
    
    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column

    # meta info
    TE_real = None
    TE_report = 'report_test.txt'
    VAL_rate = float(RD.loadArray('val_rate.txt')[0][0])
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'
    augmented = True

    if augmented == True:
        TRI = 'train_input_augmented.txt'
        TRO = 'train_output_augmented.txt'
    else:
        TRI = 'train_input.txt'
        TRO = 'train_output.txt'
    TEI = 'test_input.txt'
    TEO = 'test_predict.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epochs = int(input('epochs'))
    printed = int(input('printed? (0 -> do not print)'))

    # configurations
    times = 1
    loss = 'mse'
    opti = optimizers.Adam(0.0005, decay=1e-6)

    with K.tf.device(deviceName):

        # create model
        model = used_model.MODEL()
        model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

        # callback list for training
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=0.0001, mode='min')

        # training and test
        # 'config.txt' is used for configuration
        for i in range(times):

            toTrainLimit = 2 * 61578
            toTestLimit = 79975

            # load training input
            print('loading training input...')
            TRI_array = np.array(RD.loadArray(TRI, '\t', UTF8=False, type_='f'))
            
            if len(TRI_array) > toTrainLimit:
                TRI_array = TRI_array[:toTrainLimit]

            # load training output
            print('loading training output...')
            TRO_array = np.array(RD.loadArray(TRO, '\t', UTF8=False, type_='f'))

            if len(TRO_array) > toTrainLimit:
                TRO_array = TRO_array[:toTrainLimit]

            # load test input
            print('loading test input...')
            TEI_array = np.array(RD.loadArray(TEI, '\t', UTF8=False, type_='f'))

            if len(TEI_array) > toTestLimit:
                TEI_array = TEI_array[:toTestLimit]

            # train and save the model
            model.fit(TRI_array, TRO_array, validation_split=VAL_rate, callbacks=[early, lr_reduced], epochs=epochs)
            model.summary()
            model.save('model_e_' + str(epochs))

            # load the model
            loaded_model = tf.keras.models.load_model('model_e_' + str(epochs))

            # validation
            prediction = loaded_model.predict(TEI_array)

            # write result for validation
            if VAL_rate > 0:
                RD.saveArray('valid_prediction_' + str(i) + '.txt', prediction, '\t', 500)
            else:
                RD.saveArray('test_prediction_' + str(i) + '.txt', prediction, '\t', 500)
