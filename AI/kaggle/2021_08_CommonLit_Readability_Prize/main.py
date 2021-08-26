# ref: https://wikidocs.net/45101

import math
import numpy as np
import pandas as pd

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
    
    def __init__(self, vocab_size, max_length, embed_dim, dropout_rate=0.25, training=False, name='text_model'):
        
        super(TEXT_MODEL_LSTM, self).__init__(name=name)

        # constants
        self.vocabSize = vocab_size
        self.maxLength = max_length

        # flatten layer and regularizer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat    = tf.keras.layers.Flatten()
        L2           = tf.keras.regularizers.l2(0.001)

        # layers
        self.embedding   = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.LSTM        = tf.keras.layers.LSTM(256, name='LSTM')
        self.dense       = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=L2, name='dense')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        # Embedding, LSTM and dense
        inputs = self.embedding(inputs)
        inputs = self.LSTM(inputs)
        inputs = self.dropout(inputs, training)
        inputs = self.dense(inputs)
        inputs = self.dropout(inputs, training)
        model_output = self.dense_final(inputs)
        
        return model_output

def defineModel(vocab_size, max_length, embed_dim, dropout_rate):
    return TEXT_MODEL_LSTM(vocab_size=vocab_size,
                           max_length=max_length,
                           embed_dim=embed_dim,
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

def tokenizeInput(train_X, test_X):
    all_X = np.concatenate((train_X, test_X), axis=0)
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

def defineAndTrainModel(vocab_size, max_len, encoded_train_X, train_Y):
    # create model
    model = defineModel(vocab_size, max_len, 200, 0.25)
        
    # train/valid using model
    trainModel(model, encoded_train_X, train_Y,
               validation_split=0.1,
               epochs=10,
               early_patience=5,
               lr_reduced_factor=0.1,
               lr_reduced_patience=3)

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

if __name__ == '__main__':

    # read data
    (train_X, train_Y, test_X, ids) = loadData()

    # normalize output data
    (train_Y, avg, stddev) = normalize(train_Y)
    print('avg=' + str(round(avg, 6)))
    print('stddev=' + str(round(stddev, 6)))
    
    # tokenize input data
    t = tokenizeInput(train_X, test_X)
    vocab_size = len(t.word_index) + 1

    # encode input data
    encoded_train_X = encodeX(train_X, t)
    encoded_test_X = encodeX(test_X, t)

    # add padding to input data
    (encoded_train_X, encoded_test_X, max_len) = addPadding(encoded_train_X, encoded_test_X)
    print('max_len=' + str(max_len))
    
    # train using GPU
    try:
        # for local with GPU
        with tf.device('/gpu:0'):
            model = defineAndTrainModel(vocab_size, max_len, encoded_train_X, train_Y)
            
    except:
        # for kaggle notebook or local with no GPU
        print('cannot find GPU')
        model = defineAndTrainModel(vocab_size, max_len, encoded_train_X, train_Y)
    
    # predict using model
    prediction = model.predict(encoded_test_X).flatten()
    prediction = np.clip(prediction, 0.0001, 0.9999)

    # y -> denormalize(invSigmoid(y))
    for i in range(len(prediction)):
        prediction[i] = invSigmoid(prediction[i])
        
    prediction = prediction * stddev + avg

    # write final submission
    print('prediction:')
    print(prediction)
    
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': prediction
    })
    final_submission.to_csv('submission.csv', index=False)
