# ref: https://wikidocs.net/45101

import math
import numpy as np
import pandas as pd
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import main_helper as h
import NN_models as nns

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

INFOS = 15 # need to be changed if information extracting method changed

def loadData():
    try:
        # for local
        trainData = pd.read_csv('train.csv')
        testData = pd.read_csv('test.csv')
        
    except:
        # for kaggle notebook
        print('now running kaggle notebook')
        trainData = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
        testData = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')

    train_X = np.array(trainData['excerpt'])
    train_Y = np.array(trainData['target'])
    test_X = np.array(testData['excerpt'])

    return (train_X, train_Y, test_X, testData['id'])

def tokenizeInput(all_X):
    regex = re.compile('[^a-zA-Z ]')

    # remove excpet for a-z, A-Z and space, and all words to lowercase
    for i in range(len(all_X)):
        all_X[i] = regex.sub('', str(all_X[i]))
        all_X[i] = all_X[i].lower()

    t = Tokenizer()
    t.fit_on_texts(all_X)
    return t

def encodeX(X, tokenizer):
    encoded_X = []

    for i in range(len(X)):
        encoded = tokenizer.texts_to_sequences([X[i]])[0]
        encoded_X.append(encoded)
        
    return encoded_X

def addPadding(encoded_train_X, encoded_test_X):
    max_len_train = max(len(x) for x in encoded_train_X)
    max_len_test = max(len(x) for x in encoded_test_X)
    max_len = max(max_len_train, max_len_test)

    # add padding
    encoded_train_X = pad_sequences(encoded_train_X, maxlen=max_len, padding='pre')
    encoded_test_X = pad_sequences(encoded_test_X, maxlen=max_len, padding='pre')

    return (encoded_train_X, encoded_test_X, max_len)

def trainModel(model, X, Y, validation_split, epochs, early_patience, lr_reduced_factor, lr_reduced_patience):

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             mode="min",
                                             patience=early_patience)
    
    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=lr_reduced_factor,
                                                      patience=lr_reduced_patience,
                                                      verbose=1,
                                                      min_delta=0.0001,
                                                      mode='min')
        
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, validation_split=validation_split, callbacks=[early, lr_reduced], epochs=epochs)
    model.summary()

def sigmoid(y):
    return 1.0 / (1.0 + math.exp(-y))

def invSigmoid(y):
    return math.log(y / (1.0 - y))

def normalize(Y):
    avg = np.mean(Y)
    stddev = np.std(Y)

    # y -> sigmoid(normalize(y))
    Y = (Y - avg) / stddev

    for i in range(len(Y)):
        Y[i] = sigmoid(Y[i])
        
    return (Y, avg, stddev)

# find info : distribution of vocab index (min, 10%, median, 90%, max and average) with applying log2
def getVocabIndexDistribution(encoded_train_X, tokenizer):
    
    leng = len(encoded_train_X)
    
    # get max, min and average
    info_max = max(encoded_train_X)
    info_min = min(encoded_train_X)
    info_avg = np.mean(encoded_train_X)

    # get 1%, 5%, 10%, 30%, median, 70%, 90%, 95% and 99%
    leng_01p = int(0.01 * leng)
    leng_05p = int(0.05 * leng)
    leng_10p = int(0.1  * leng)
    leng_30p = int(0.3  * leng)
    leng_med = int(0.5  * leng)
    leng_70p = int(0.7  * leng)
    leng_90p = int(0.9  * leng)
    leng_95p = int(0.95 * leng)
    leng_99p = int(0.99 * leng)

    info_01p = np.partition(encoded_train_X, leng_01p)[leng_01p]
    info_05p = np.partition(encoded_train_X, leng_05p)[leng_05p]
    info_10p = np.partition(encoded_train_X, leng_10p)[leng_10p]
    info_30p = np.partition(encoded_train_X, leng_30p)[leng_30p]
    info_med = np.partition(encoded_train_X, leng_med)[leng_med]
    info_70p = np.partition(encoded_train_X, leng_70p)[leng_70p]
    info_90p = np.partition(encoded_train_X, leng_90p)[leng_90p]
    info_95p = np.partition(encoded_train_X, leng_95p)[leng_95p]
    info_99p = np.partition(encoded_train_X, leng_99p)[leng_99p]
    
    # return the info
    return np.array([math.log(info_min),
                     math.log(info_max),
                     math.log(info_avg),
                     math.log(info_01p), math.log(info_05p), math.log(info_10p),
                     math.log(info_30p), math.log(info_med), math.log(info_70p),
                     math.log(info_90p), math.log(info_95p), math.log(info_99p)])

# find info : number of sentences, average sentence length (in words), and total length (in words)
def getSentenceInfo(sentence):

    numberOfSentences = len(sentence.split('.'))
    totalLength       = len(sentence.split(' '))
    avgLength         = totalLength / numberOfSentences

    # return the info
    return np.array([numberOfSentences, totalLength, avgLength])

# find additional info
def getAdditionalInfo(pad_encoded_X, leng, X, encoded_X):

    additionalInfo = np.zeros((leng, INFOS))

    for i in range(leng):

        # add { distribution of vocab index (min, 10%, median, 90%, max and average) } with applying log2
        # to encoded train_X
        vocabDist = getVocabIndexDistribution(encoded_X[i], t)
        
        # add { number of sentences, average sentence length (in words), total length (in words) }
        # to encoded train_X
        sentenceInfo = getSentenceInfo(X[i])
        
        additionalInfo[i] = np.concatenate((vocabDist, sentenceInfo), axis=0)

    # normalize additional info
    for col in range(INFOS):
        (additionalInfo[:, col], avg, stddev) = normalize(additionalInfo[:, col])
    
    pad_encoded_X = np.concatenate((pad_encoded_X, additionalInfo), axis=1)

    return pad_encoded_X

if __name__ == '__main__':

    # read data
    (train_X, train_Y, test_X, ids) = loadData()

    trainLen = len(train_X)
    testLen = len(test_X)
    totalLen = trainLen + testLen
    all_X = np.concatenate((train_X, test_X), axis=0)

    # normalize output data
    (train_Y, avg, stddev) = normalize(train_Y)
    print('avg=' + str(round(avg, 6)))
    print('stddev=' + str(round(stddev, 6)))

    # save normalization info
    normalization = pd.DataFrame([[avg, stddev]])
    normalization.columns = ['avg', 'stddev']
    normalization.to_csv('normalization_info.csv')
    
    # tokenize input data
    t = tokenizeInput(all_X)
    vocab_size = len(t.word_index) + 1

    # encode input data
    encoded_train_X = encodeX(train_X, t)
    encoded_test_X = encodeX(test_X, t)

    # add padding to input data
    (pad_encoded_train_X, pad_encoded_test_X, max_len) = addPadding(encoded_train_X, encoded_test_X)
    print('max_len=' + str(max_len))

    # find additional info
    pad_encoded_train_X = getAdditionalInfo(pad_encoded_train_X, trainLen, train_X, encoded_train_X)
    pad_encoded_test_X = getAdditionalInfo(pad_encoded_test_X, testLen, test_X, encoded_test_X)
    
    # train/valid using GPU
    valid_rate = 0.15

    train_rows = int(trainLen * (1.0 - valid_rate))
    pad_encoded_train_X_train = pad_encoded_train_X[:train_rows]
    pad_encoded_train_Y_train = train_Y[:train_rows]
    pad_encoded_train_X_valid = pad_encoded_train_X[train_rows:]
    pad_encoded_train_Y_valid = train_Y[train_rows:]

    # save original validation data
    pad_encoded_train_Y_valid_invSigmoid = np.log(pad_encoded_train_Y_valid / (1.0 - pad_encoded_train_Y_valid))
    pd.DataFrame(pad_encoded_train_Y_valid_invSigmoid).to_csv('valid_original.csv')

    # print data
    print('\n[ pad_encoded_train_X_train ]')
    print(np.shape(pad_encoded_train_X_train))
    print(np.array(pad_encoded_train_X_train))
    
    print('\n[ pad_encoded_train_Y_train ]')
    print(np.shape(pad_encoded_train_Y_train))
    print('mean:', np.mean(pad_encoded_train_Y_train))
    print(np.array(pad_encoded_train_Y_train[:100]))
    
    print('\n[ pad_encoded_train_X_valid ]')
    print(np.shape(pad_encoded_train_X_valid))
    print(np.array(pad_encoded_train_X_valid))

    print('\n[ pad_encoded_train_Y_valid ]')
    print(np.shape(pad_encoded_train_Y_valid))
    print('mean:', np.mean(pad_encoded_train_Y_valid))
    print(np.array(pad_encoded_train_Y_valid[:100]))

    # DEEP LEARNING
    model_type  = [1, 1, 1, 1, 1, 1, 1] # TEXT_MODEL_LSTM1 only
    embed_dims  = [200, 200, 200, 200, 200, 200, -1]
    cnn_filters = [32, 64, 128, 32, 64, 128, -1]
    use_info    = [True, True, True, False, False, False, True]
    
    use_at_once = 10 # first 10 tokens with size 768 each

    for i in range(len(use_info)):

        # define and train model
        with tf.device('/gpu:0'):

            print('No:', i, 'use_INFO:', use_info[i], 'use_LSTM:', cnn_filters[i] >= 0)

            # define model (error occurs now)

            # TEXT_MODEL_LSTM1          
            model = nns.TEXT_MODEL_LSTM1(vocab_size, max_len,
                                         embed_dim=embed_dims[i], cnn_filters=cnn_filters[i],
                                         use_INFO=use_info[i], use_LSTM=(cnn_filters[i] >= 0), INFOS=15)

            # train model using GPU
            h.trainModel(model, pad_encoded_train_X_train, pad_encoded_train_Y_train, valid_rate,
                         epochs=15, early_patience=4, lr_reduced_factor=0.1, lr_reduced_patience=2)

            # save model
            model.save('main_LSTM1_' + str(i))

        # valid and test using GPU
        valid_prediction = model.predict(pad_encoded_train_X_valid)
        test_prediction  = model.predict(pad_encoded_test_X)

        # invSigmoid
        valid_prediction = np.log(valid_prediction / (1.0 - valid_prediction))
        test_prediction  = np.log(test_prediction  / (1.0 - test_prediction))

        pd.DataFrame(valid_prediction).to_csv('valid_LSTM1_' + str(i) + '.csv')
        pd.DataFrame(test_prediction).to_csv('test_LSTM1_' + str(i) + '.csv')
