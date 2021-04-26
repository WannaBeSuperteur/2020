# train_train.csv, train_valid.csv -> bert_train_result.txt, bert_valid_result.txt
# additional need: bert_max_lengths_train.txt

# ref: https://insightcampus.co.kr/insightcommunity/?mod=document&uid=12854
#      https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
#      https://www.kaggle.com/linwenxing/glove-vector-lstm

import sys
sys.path.insert(0, '../../../AI_BASE')
sys.path.insert(1, '../')
import my_extract as ME

import math
import numpy as np
import readData as RD
import deepLearning_main as DL
import pandas as pd
import random
import gc

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import bert

def tokenize_text(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self, vocabulary_size,
                 embedding_dimensions=128, cnn_filters=50, dnn_units=512,
                 dropout_rate=0.1, training=False, name='text_model', useState=True, max_length=0):
        
        super(TEXT_MODEL, self).__init__(name=name)

        # constants
        # u00 : for text (max_length), u01 : for state (51)
        self.u_text = max_length
        self.d_t = 100
        self.output_length = 24

        if useState == True:
            self.d_s0, self.d_s1, self.d_s2, self.d_sf = 100, 100, 100, 100
            self.u_state = 51

        # L2 regularization
        L2 = tf.keras.regularizers.l2(0.001)

        # for state
        if useState == True:
            self.den_state_0 = tf.keras.layers.Dense(units=self.d_s0, activation='relu', kernel_regularizer=L2, name='BERT_dense')
            self.den_state_1 = tf.keras.layers.Dense(units=self.d_s1, activation='relu', kernel_regularizer=L2, name='BERT_dense')
            self.den_state_2 = tf.keras.layers.Dense(units=self.d_s2, activation='relu', kernel_regularizer=L2, name='BERT_dense')
            self.den_state_final = tf.keras.layers.Dense(units=self.d_sf, activation='relu', kernel_regularizer=L2, name='BERT_dense')

        # for BERT-tokenized text
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='BERT_embedding')
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding='valid', activation='relu', name='BERT_cnn1')
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu', name='BERT_cnn2')
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu', name='BERT_cnn3')
        self.pool = tf.keras.layers.GlobalMaxPool1D(name='BERT_pooling')
        self.den_text = tf.keras.layers.Dense(units=self.d_t, activation='relu', kernel_regularizer=L2, name='BERT_dense')

        # for concatenated layer and output
        self.dense = tf.keras.layers.Dense(units=dnn_units, activation='relu', kernel_regularizer=L2, name='FINAL_dense0')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.last_dense = tf.keras.layers.Dense(units=self.output_length, activation='sigmoid', name='output') # output layer

    def call(self, inputs, training):

        # split input
        if useState == True:
            i_text, i_state = tf.split(inputs, [self.u_text, self.u_state], 1)
        else:
            i_text = inputs
            
        i_text = tf.cast(i_text, tf.int32)

        # info: state
        if useState == True:
            i_state_den0 = self.den_state_0(i_state)
            i_state_den1 = self.den_state_1(i_state_den0)
            i_state_den2 = self.den_state_2(i_state_den1)
            i_state_denFinal = self.den_state_final(i_state_den2)
            i_state_drop = self.dropout(i_state_denFinal, training)

        # BERT-tokenized text
        i_text_emb = self.embedding(i_text)
        
        i_text_emb_1 = self.cnn_layer1(i_text_emb) 
        i_text_emb_1 = self.pool(i_text_emb_1) 
        i_text_emb_2 = self.cnn_layer2(i_text_emb) 
        i_text_emb_2 = self.pool(i_text_emb_2)
        i_text_emb_3 = self.cnn_layer3(i_text_emb)
        i_text_emb_3 = self.pool(i_text_emb_3)
        
        i_text_concat = tf.concat([i_text_emb_1, i_text_emb_2, i_text_emb_3], axis=-1)
        i_text_den = self.den_text(i_text_concat)
        i_text_drop = self.dropout(i_text_den, training)

        # concatenate embedded info and concatenated BERT-tokenized text
        if useState == True:
            concatenated = tf.concat([i_text_drop, i_state_drop], axis=-1)
        else:
            concatenated = i_text_drop
        
        concatenated = self.dense(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output

def convertForBert(input_data, print_interval, tokenizer, max_length):

    rows = len(input_data)
    
    # convert training/valid input data into the list of word IDs
    tokenized_input = []

    new_max_length = 0
        
    for i in range(rows):
        if i % print_interval == 0: print(i)

        tokenized_txt = tokenize_text(input_data[i], tokenizer)

        # add each element of tokenized text
        this_tokenized_txt = []
            
        for j in range(len(tokenized_txt)):
            this_tokenized_txt.append(tokenized_txt[j])

        # update new max_length
        if max_length == None:
            if len(this_tokenized_txt) > new_max_length:
                new_max_length = len(this_tokenized_txt)

        # insert the tokenized text to input list
        tokenized_input.append(this_tokenized_txt)

    # return new max_length if max_length is null
    if max_length == None:
        return new_max_length

    # append 0 so that the length is all the same for each tokenized input
    for i in range(rows):
        while len(tokenized_input[i]) < max_length:
            tokenized_input[i].append(0)

    # for training
    try:
        print('\n[c01] shape of tokenized input:')
        print(np.shape(tokenized_input))
        print('------')
        print(np.array(tokenized_input))

        return tokenized_input

    # for valid
    except:
        return tokenized_input

def createBatchedDataset(processed_dataset, rows, batch_size):
    batched_dataset = processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))
        
    total_batches = math.ceil(rows / batch_size)
    batched_dataset.shuffle(total_batches)

    return batched_dataset

if __name__ == '__main__':

    # create bert tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # configuration
    train_rows = 66253
    valid_rows = 11692
    print_interval = 500
    batch_size = 32

    embedding_dim = 200
    cnn_filters = 50
    dnn_units = 256
    dropout_rate = 0.3

    loss = 'mse'
    opti = optimizers.Adam(0.0005, decay=1e-6)

    epochs = 15
    batch_size = 32
    useState = False

    # max length of all the train/valid tokenized inputs
    max_length = 71

    # create text model
    text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab),
                            embedding_dimensions=embedding_dim,
                            cnn_filters=cnn_filters,
                            dnn_units=dnn_units, dropout_rate=dropout_rate,
                            useState=useState, max_length=max_length)

    text_model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

    # load training and valid/test data
    print('loading training input...')
    train_input = RD.loadArray('train_input.txt')

    print('loading training output...')
    train_output = np.array(RD.loadArray('train_output.txt')).astype(float)

    print('loading valid input...')
    valid_input = RD.loadArray('test_input.txt')

    print('loading valid output...')
    valid_output = []
    
    print(' *----- data -----*')
    print(np.shape(train_input))
    print(np.array(train_input))
    print(np.shape(train_output))
    print(np.array(train_output))
    print(np.shape(valid_input))
    print(np.array(valid_input))
    print(np.shape(valid_output))
    print(np.array(valid_output))
    
    # create train input and valid input
    train_text = np.array(train_input)[:, 0].astype(str)
    train_tokenized = np.array(convertForBert(train_text, print_interval, tokenizer, max_length)).astype(float)

    if useState == True:
        train_state = np.array(train_input)[:, 1:].astype(float)

    print(' *----- train -----*')
    print(np.shape(train_text))
    print(train_text)
    print(np.shape(train_tokenized))
    print(train_tokenized)

    if useState == True:
        print(np.shape(train_state))
        print(train_state)

    if useState == True:
        final_train_input = np.concatenate((train_tokenized, train_state), axis=1).astype(float)
    else:
        final_train_input = train_tokenized.astype(float)

    valid_text = np.array(valid_input)[:, 0].astype(str)
    valid_tokenized = np.array(convertForBert(valid_text, print_interval, tokenizer, max_length)).astype(float)

    if useState == True:
        valid_state = np.array(valid_input)[:, 1:].astype(float)

    print(' *----- valid -----*')
    print(np.shape(valid_text))
    print(valid_text)
    print(np.shape(valid_tokenized))
    print(valid_tokenized)

    if useState == True:
        print(np.shape(valid_state))
        print(valid_state)

    if useState == True:
        final_valid_input = np.concatenate((valid_tokenized, valid_state), axis=1).astype(float)
    else:
        final_valid_input = valid_tokenized.astype(float)

    RD.saveArray('train_tokenized.txt', train_tokenized, '\t', 500)
    RD.saveArray('valid_tokenized.txt', valid_tokenized, '\t', 500)

    print(' *----- final -----*')
    print(np.shape(final_train_input))
    print(final_train_input)
    print(np.shape(train_output))
    print(train_output)
    print(np.shape(final_valid_input))
    print(final_valid_input)
    print(np.shape(valid_output))
    print(valid_output)

    # callback list for training
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=0.0001, mode='min')

    # train and save the model
    text_model.fit(final_train_input, train_output, validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)
    text_model.summary()
    text_model.save('model_e_' + str(epochs))

    # load the model
    loaded_model = tf.keras.models.load_model('model_e_' + str(epochs))

    # validation
    prediction = loaded_model.predict(final_valid_input)

    # write result for validation
    RD.saveArray('bert_valid_result.txt', prediction, '\t', 500)
