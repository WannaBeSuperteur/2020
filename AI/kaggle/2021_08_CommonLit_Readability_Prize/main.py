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

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

class TEXT_MODEL_LSTM(tf.keras.Model):
    
    def __init__(self, vocab_size, max_length, embed_dim, cnn_filters, dropout_rate=0.25, training=False, name='text_model'):
        
        super(TEXT_MODEL_LSTM, self).__init__(name=name)

        # constants
        self.vocabSize = vocab_size
        self.maxLength = max_length

        # flatten layer and regularizer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat    = tf.keras.layers.Flatten()
        L2           = tf.keras.regularizers.l2(0.001)

        # use CNN (CNN2, CNN3, CNN4) + original input
        self.CNN2    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding='valid', activation='relu', name='CNN2_text')
        self.CNN3    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu', name='CNN3_text')
        self.CNN4    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu', name='CNN4_text')

        # layers for inputs_text
        self.embedding   = tf.keras.layers.Embedding(vocab_size, embed_dim, name='embedding_text')
        self.LSTM        = tf.keras.layers.LSTM(64, name='LSTM_text')
        self.dense_text  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_text')

        # layers for inputs_info
        self.dense_info  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_info')

        # layers for final output
        self.merged      = tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=L2, name='merged')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        # split inputs
        max_len = self.maxLength
        inputs_text, unused, inputs_info = tf.split(inputs, [max_len-1, 1, 9], axis=1)

        # CNN layers
        inputs_text = tf.reshape(inputs_text, (-1, max_len-1))

        # Embedding, LSTM, CNN and dense for inputs_text
        inputs_text = self.embedding(inputs_text)
        inputs_text = self.dropout(inputs_text, training)
        
        inputs_text = self.LSTM(inputs_text)
        inputs_text = self.dropout(inputs_text, training)

        # use CNN
        inputs_text = tf.reshape(inputs_text, (-1, 64, 1))
        
        inputs_text_CNN2 = self.CNN2(inputs_text)
        inputs_text_CNN3 = self.CNN3(inputs_text)
        inputs_text_CNN4 = self.CNN4(inputs_text)

        inputs_text_CNN2 = self.flat(inputs_text_CNN2)
        inputs_text_CNN3 = self.flat(inputs_text_CNN3)
        inputs_text_CNN4 = self.flat(inputs_text_CNN4)
        
        inputs_text = tf.concat([inputs_text_CNN2, inputs_text_CNN3, inputs_text_CNN4], axis=-1)
        inputs_text = self.dropout(inputs_text, training)

        # dense after CNN
        inputs_text = self.dense_text(inputs_text)
        inputs_text = self.dropout(inputs_text, training)
        
        # DNN for inputs_info
        inputs_info = self.dense_info(inputs_info)
        inputs_info = self.dropout(inputs_info, training)
        
        # concatenate inputs_text and inputs_info
        concatenated = tf.concat([inputs_text, inputs_info], axis=-1)

        # final output
        model_merged = self.merged(concatenated)
        model_output = self.dense_final(model_merged)
        
        return model_output

def defineModel(vocab_size, max_length, embed_dim, cnn_filters, dropout_rate):
    return TEXT_MODEL_LSTM(vocab_size=vocab_size,
                           max_length=max_length,
                           embed_dim=embed_dim,
                           cnn_filters=cnn_filters,
                           dropout_rate=dropout_rate)

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

def defineAndTrainModel(vocab_size, max_len, pad_encoded_train_X, train_Y):
    
    # create model
    embed_dim    = 64
    cnn_filters  = 64
    dropout_rate = 0.25
    
    model = defineModel(vocab_size, max_len, embed_dim, cnn_filters, dropout_rate)

    # model configuration
    val_split           = 0.1
    epochs              = 100
    early_patience      = 6
    lr_reduced_factor   = 0.1
    lr_reduced_patience = 2
        
    # train/valid using model
    trainModel(model, pad_encoded_train_X, train_Y,
               validation_split=val_split,
               epochs=epochs,
               early_patience=early_patience,
               lr_reduced_factor=lr_reduced_factor,
               lr_reduced_patience=lr_reduced_patience)

    return model

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

    # get 10%, median and 90%
    leng_10p = int(0.1 * leng)
    leng_med = int(0.5 * leng)
    leng_90p = int(0.9 * leng)
    
    info_10p = np.partition(encoded_train_X, leng_10p)[leng_10p]
    info_med = np.partition(encoded_train_X, leng_med)[leng_med]
    info_90p = np.partition(encoded_train_X, leng_90p)[leng_90p]

    # return the info
    return np.array([math.log(info_min),
                     math.log(info_10p),
                     math.log(info_med),
                     math.log(info_90p),
                     math.log(info_max),
                     math.log(info_avg)])

# find info : number of sentences, average sentence length (in words), and total length (in words)
def getSentenceInfo(sentence):

    numberOfSentences = len(sentence.split('.'))
    totalLength       = len(sentence.split(' '))
    avgLength         = totalLength / numberOfSentences

    # return the info
    return np.array([numberOfSentences, totalLength, avgLength])

# find additional info
def getAdditionalInfo(pad_encoded_X, leng, X, encoded_X):

    additionalInfo = np.zeros((leng, 9))

    for i in range(leng):

        # add { distribution of vocab index (min, 10%, median, 90%, max and average) } with applying log2
        # to encoded train_X
        vocabDist = getVocabIndexDistribution(encoded_X[i], t)

        # add { number of sentences, average sentence length (in words), total length (in words) }
        # to encoded train_X
        sentenceInfo = getSentenceInfo(X[i])

        additionalInfo[i] = np.concatenate((vocabDist, sentenceInfo), axis=0)

    # normalize additional info
    for col in range(9):
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
    h.trainOrValid(True, 'main', valid_rate, trainLen, max_lens, [pad_encoded_train_X], train_Y, [pad_encoded_test_X],
                   count, avg, stddev)

    # test using GPU
    final_prediction = h.trainOrValid(False, 'main', valid_rate, trainLen, max_lens, [pad_encoded_train_X], train_Y, [pad_encoded_test_X],
                                      count, avg, stddev)
    
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': prediction
    })
    final_submission.to_csv('submission.csv', index=False)
