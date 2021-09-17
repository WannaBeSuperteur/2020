# train_train.csv, train_valid.csv -> bert_train_result.txt, bert_valid_result.txt
# additional need: bert_max_lengths_train.txt

# ref: https://insightcampus.co.kr/insightcommunity/?mod=document&uid=12854
#      https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
#      https://www.kaggle.com/linwenxing/glove-vector-lstm

import math
import numpy as np
import pandas as pd
import random
import gc

from transformers import DistilBertTokenizer, DistilBertModel
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

#import bert
from bert.tokenization import bert_tokenization

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self, vocabulary_size,
                 embedding_dimensions=256, cnn_filters=50, dnn_units=256,
                 dropout_rate=0.25,
                 max_tokens=np.array([[1], [2], [3], [4], [5], [6]]),
                 training=False, name='text_model'):
        
        super(TEXT_MODEL, self).__init__(name=name)

        # flatten layer
        self.flat = tf.keras.layers.Flatten()
        self.maxtokens = max_tokens
        self.d00, self.d01, self.d02, self.d03, self.d04, self.d05 = 20, 20, 20, 20, 20, 20

        # for info 0, 1, 2, 3, 4, 5, 6, 7, 8 and i9_cont
        L2 = tf.keras.regularizers.l2(0.001)
        
        self.emb00 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb00')
        self.emb01 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb01')
        self.emb02 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb02')
        self.emb03 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb03')
        self.emb04 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb04')
        self.emb05 = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='emb05')
        
        self.den00 = tf.keras.layers.Dense(units=self.d00, activation='relu', kernel_regularizer=L2, name='den00')
        self.den01 = tf.keras.layers.Dense(units=self.d01, activation='relu', kernel_regularizer=L2, name='den01')
        self.den02 = tf.keras.layers.Dense(units=self.d02, activation='relu', kernel_regularizer=L2, name='den02')
        self.den03 = tf.keras.layers.Dense(units=self.d03, activation='relu', kernel_regularizer=L2, name='den03')
        self.den04 = tf.keras.layers.Dense(units=self.d04, activation='relu', kernel_regularizer=L2, name='den04')
        self.den05 = tf.keras.layers.Dense(units=self.d05, activation='relu', kernel_regularizer=L2, name='den05')
        
        # for BERT-tokenized text
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='BERT_embedding')

        # for concatenated layer and output
        self.dense = tf.keras.layers.Dense(units=dnn_units, activation='relu', kernel_regularizer=L2, name='FINAL_dense0')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.last_dense = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output') # output layer

    def call(self, inputs, training):

        # split input
        maxtokens = self.maxtokens
        
        title, essay1, essay2, essay3, essay4, summary = tf.split(inputs,
                                                                  [maxtokens[0][0], maxtokens[1][0], maxtokens[2][0],
                                                                   maxtokens[3][0], maxtokens[4][0], maxtokens[5][0]], 1)

        title = tf.cast(title, tf.int32)
        essay1 = tf.cast(essay1, tf.int32)
        essay2 = tf.cast(essay2, tf.int32)
        essay3 = tf.cast(essay3, tf.int32)
        essay4 = tf.cast(essay4, tf.int32)
        summary = tf.cast(summary, tf.int32)

        title = self.emb00(title)
        title = self.dropout(title, training)
        title = self.flat(title)
        title = self.den00(title)
        title = self.dropout(title, training)
        
        essay1 = self.emb01(essay1)
        essay1 = self.dropout(essay1, training)
        essay1 = self.flat(essay1)
        essay1 = self.den01(essay1)
        essay1 = self.dropout(essay1, training)
        
        essay2 = self.emb02(essay2)
        essay2 = self.dropout(essay2, training)
        essay2 = self.flat(essay2)
        essay2 = self.den02(essay2)
        essay2 = self.dropout(essay2, training)
        
        essay3 = self.emb03(essay3)
        essay3 = self.dropout(essay3, training)
        essay3 = self.flat(essay3)
        essay3 = self.den03(essay3)
        essay3 = self.dropout(essay3, training)
        
        essay4 = self.emb04(essay4)
        essay4 = self.dropout(essay4, training)
        essay4 = self.flat(essay4)
        essay4 = self.den04(essay4)
        essay4 = self.dropout(essay4, training)

        summary = self.emb05(summary)
        summary = self.dropout(summary, training)
        summary = self.flat(summary)
        summary = self.den05(summary)
        summary = self.dropout(summary, training)

        # concatenate embedded info and concatenated BERT-tokenized text
        concatenated = tf.concat([title, essay1, essay2, essay3, essay4, summary], axis=-1)
        
        concatenated = self.dense(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output

def main(model, tokenizer):
    print(len(tokenizer.vocab))

    # define configuration
    train_max_rows = 9999999
    valid_max_rows = 9999999
    print_interval = 160

    embedding_dim = 200
    cnn_filters = 50
    dnn_units = 256
    dropout_rate = 0.5

    loss = 'mse'
    opti = optimizers.Adam(0.0005, decay=1e-6)
    epochs = 15

    trainInput = 'train_input_text.csv'
    trainOutput = 'train_output.csv'
    validInput = 'test_input_text.csv'
    
    maxTokens = 'DistilBert_max_tokens.csv'
    deviceName = 'gpu:0'

    # load training and valid data
    train_input = np.array(pd.read_csv(trainInput, index_col=0))
    train_output = np.array(pd.read_csv(trainOutput, index_col=0))
    valid_input = np.array(pd.read_csv(validInput, index_col=0))

    # print the number of rows to train/valid
    rows_to_train = min(len(train_input), train_max_rows)
    rows_to_valid = min(len(valid_input), valid_max_rows)
    print('\n[00] rows to train : ' + str(rows_to_train))
    print('\n[01] rows to valid : ' + str(rows_to_valid))

    # load max token count
    max_tokens = np.array(pd.read_csv(maxTokens, index_col=0)) # shape: (6, 1)
    
    # train
    with tf.device(deviceName):

        # load model
        try:
            loaded_model = tf.keras.models.load_model('DistilBert_based_model')

        except:
            text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab),
                                    embedding_dimensions=embedding_dim,
                                    cnn_filters=cnn_filters,
                                    dnn_units=dnn_units, dropout_rate=dropout_rate,
                                    max_tokens=max_tokens)

            text_model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

            # callback list for training
            early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3)
            lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=0.0001, mode='min')

            # train the model
            text_model.fit(train_input, train_output, validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)
            text_model.summary()

            # save the model
            text_model.save('DistilBert_based_model')

            # load the model
            loaded_model = tf.keras.models.load_model('DistilBert_based_model')

        # validation
        print('predicting...')
        prediction = loaded_model.predict(valid_input)

        print('\n[02] prediction')
        print(np.shape(prediction))
        print(np.array(prediction))

        prediction = pd.DataFrame(prediction)
        prediction.to_csv('DistilBert_prediction.csv')

if __name__ == '__main__':

    # load DistilBert tokenizer and model
    DistilBert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    DistilBert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    main(DistilBert_model, DistilBert_tokenizer)