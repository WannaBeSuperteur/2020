# ref: https://wikidocs.net/45101
#      https://huggingface.co/transformers/model_doc/roberta.html

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

from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

# model
class TEXT_MODEL_LSTM(tf.keras.Model):
    
    def __init__(self, max_length, embed_dim, cnn_filters, dropout_rate=0.25, training=False, name='text_model'):
        
        super(TEXT_MODEL_LSTM, self).__init__(name=name)

        # constants
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
        self.dense_text  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_text')

        # layers for inputs_info
        self.dense_info  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_info')

        # layers for final output
        self.merged      = tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=L2, name='merged')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        ### split inputs
        max_len = self.maxLength
        inputs_text, inputs_info = tf.split(inputs, [max_len, 3], axis=1)

        ### for text input
        # CNN layers
        inputs_text = tf.reshape(inputs_text, (-1, max_len, 1))

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

        ### for info input
        # DNN for inputs_info
        inputs_info = self.dense_info(inputs_info)
        inputs_info = self.dropout(inputs_info, training)
        
        ### concatenate inputs_text and inputs_info
        concatenated = tf.concat([inputs_text, inputs_info], axis=-1)

        # final output
        model_merged = self.merged(concatenated)
        model_output = self.dense_final(model_merged)
        
        return model_output

# load the dataset
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

# apply regular expression to the text
def applyRegex(all_X):
    regex = re.compile('[^a-zA-Z0-9.,?! ]')

    # remove excpet for a-z, A-Z, 0-9, '.', ',' and space
    for i in range(len(all_X)):
        all_X[i] = regex.sub('', str(all_X[i]))

    return all_X

# encode and add padding to the text
def encodeX(X, roberta_tokenizer, roberta_model):
    encoded_X = []

    print('')
    
    for i in range(len(X)):
        roberta_input  = roberta_tokenizer(X[i], return_tensors="tf")
        roberta_output = roberta_model(roberta_input)
        roberta_lhs    = roberta_output.last_hidden_state # shape: (1, N, 768)
        roberta_lhs    = np.array(roberta_lhs)[0] # shape: (N, 768)

        encoded = []
        for j in range(len(roberta_lhs)): # N iterations
            encoded.append(roberta_lhs[j][0])
        encoded_X.append(encoded)

        print(str(i) + ' / ' + str(len(X)) + ' : len=' + str(len(encoded)))

    print('')
        
    return encoded_X

def addPadding(encoded_train_X, encoded_test_X):
    max_len_train = max(len(x) for x in encoded_train_X)
    max_len_test = max(len(x) for x in encoded_test_X)
    max_len = max(max_len_train, max_len_test)

    # add padding
    new_encoded_train_X = np.zeros((len(encoded_train_X), max_len))
    new_encoded_test_X  = np.zeros((len(encoded_test_X) , max_len))

    for i in range(len(new_encoded_train_X)):
        new_encoded_train_X[i][max_len - len(encoded_train_X[i]):] = encoded_train_X[i]
    for i in range(len(new_encoded_test_X)):
        new_encoded_test_X [i][max_len - len(encoded_test_X[i]) :] = encoded_test_X [i]

    return (new_encoded_train_X, new_encoded_test_X, max_len)

# define the model
def defineModel(max_length, embed_dim, cnn_filters, dropout_rate):
    return TEXT_MODEL_LSTM(max_length=max_length,
                           embed_dim=embed_dim,
                           cnn_filters=cnn_filters,
                           dropout_rate=dropout_rate)

# train the model
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

def defineAndTrainModel(max_len, pad_encoded_train_X, train_Y):
    
    # create model
    embed_dim    = 64
    cnn_filters  = 64
    dropout_rate = 0.25
    
    model = defineModel(max_len, embed_dim, cnn_filters, dropout_rate)

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

# find info : number of sentences, average sentence length (in words), and total length (in words)
def getSentenceInfo(sentence):

    numberOfSentences = len(sentence.split('.'))
    totalLength       = len(sentence.split(' '))
    avgLength         = totalLength / numberOfSentences

    # return the info
    return np.array([numberOfSentences, totalLength, avgLength])

# find additional info
def getAdditionalInfo(pad_encoded_X, leng, X, encoded_X):

    additionalInfo = np.zeros((leng, 3))

    for i in range(leng):

        # add { number of sentences, average sentence length (in words), total length (in words) }
        # to encoded train_X
        additionalInfo[i] = getSentenceInfo(X[i])

    # normalize additional info
    for col in range(3):
        (additionalInfo[:, col], avg, stddev) = normalize(additionalInfo[:, col])
    
    pad_encoded_X = np.concatenate((pad_encoded_X, additionalInfo), axis=1)

    return pad_encoded_X

if __name__ == '__main__':

    # read data
    (train_X, train_Y, test_X, ids) = loadData()

    trainLen = len(train_X)
    testLen = len(test_X)

    print('\n[00] original train_Y :')
    print(np.shape(train_Y))
    print(train_Y)

    # roBERTa tokenizer and model
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    
    # normalize output data : y -> sigmoid(normalize(y))
    (train_Y, avg, stddev) = normalize(train_Y)
    print('\n[01] avg=' + str(round(avg, 6)))
    print('\n[02] stddev=' + str(round(stddev, 6)))

    print('\n[03] converted train_Y :')
    print(np.shape(train_Y))
    print(train_Y)
    
    # apply regular expression and tokenize input data
    train_X = applyRegex(train_X)
    test_X  = applyRegex(test_X)

    # encode input data
    encoded_train_X = encodeX(train_X, roberta_tokenizer, roberta_model)
    encoded_test_X  = encodeX(test_X , roberta_tokenizer, roberta_model)

    print('\n[04] encoded_train_X :')
    print(np.shape(encoded_train_X))
    
    print('\n[05] encoded_test_X :')
    print(np.shape(encoded_test_X))
    
    # add padding to input data
    (pad_encoded_train_X, pad_encoded_test_X, max_len) = addPadding(encoded_train_X, encoded_test_X)
    print('\n[06] max_len=' + str(max_len))

    print('\n[07] pad_encoded_train_X :')
    print(np.shape(pad_encoded_train_X))
    print(pad_encoded_train_X)
    
    print('\n[08] pad_encoded_test_X :')
    print(np.shape(pad_encoded_test_X))
    print(pad_encoded_test_X)
    
    # find additional info
    pad_encoded_train_X = getAdditionalInfo(pad_encoded_train_X, trainLen, train_X, encoded_train_X)
    pad_encoded_test_X = getAdditionalInfo(pad_encoded_test_X, testLen, test_X, encoded_test_X)

    print('\n[09] pad_encoded_train_X with additional info :')
    print(np.shape(pad_encoded_train_X))
    print(pad_encoded_train_X)

    print('\n[10] pad_encoded_test_X with additional info :')
    print(np.shape(pad_encoded_test_X))
    print(pad_encoded_test_X)
    
    # train using GPU
    try:
        # for local with GPU
        with tf.device('/gpu:0'):
            model = defineAndTrainModel(max_len, pad_encoded_train_X, train_Y)
            
    except:
        # for kaggle notebook or local with no GPU
        print('cannot find GPU')
        model = defineAndTrainModel(max_len, pad_encoded_train_X, train_Y)
    
    # predict using model
    prediction = model.predict(pad_encoded_test_X).flatten()
    prediction = np.clip(prediction, 0.0001, 0.9999)

    print('\n[11] original (normalized and then SIGMOID-ed) prediction:')
    print(np.shape(prediction))
    print(prediction)

    # y -> denormalize(invSigmoid(y))
    for i in range(len(prediction)):
        prediction[i] = invSigmoid(prediction[i])
        
    prediction = prediction * stddev + avg

    # write final submission
    print('\n[12] converted FINAL prediction:')
    print(np.shape(prediction))
    print(prediction)
    
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': prediction
    })
    final_submission.to_csv('submission.csv', index=False)
