# train_train.csv, train_valid.csv -> bert_train_result.txt, bert_valid_result.txt
# additional need: bert_max_lengths_train.txt

# ref: https://insightcampus.co.kr/insightcommunity/?mod=document&uid=12854
#      https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

import sys
sys.path.insert(0, '../../../AI_BASE')

import math
import numpy as np
import readData as RD
import deepLearning_main as DL
import pandas as pd
import random

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import bert

def tokenize_text(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self, vocabulary_size, embedding_dimensions=128, cnn_filters=50,
                 dnn_units=512, dropout_rate=0.1, training=False, name='text_model'):
        
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions)
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding='valid', activation='relu')
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu')
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu')
        self.pool = tf.keras.layers.GlobalMaxPool1D()
        self.dense_1 = tf.keras.layers.Dense(units=dnn_units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.last_dense = tf.keras.layers.Dense(units=1, activation='sigmoid') # output layer

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output

def convertForBert(input_data, output_data, print_interval, tokenizer, max_length):

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

        # bind input and output data
        in_out = [[input_, output_data[i]] for i, input_ in enumerate(tokenized_input)]
        in_out_labels = [(row[0], row[1]) for row in in_out]
        
        processed_dataset = tf.data.Dataset.from_generator(lambda: in_out_labels, output_types=(tf.int32, tf.int32))
        return processed_dataset

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

    # define configuration
    train_max_rows = 16000 # 9999999 (no limit)
    valid_max_rows = 4000 # 9999999 (no limit)
    print_interval = 400
    batch_size = 32

    embedding_dim = 200
    cnn_filters = 50
    dnn_units = 256
    dropout_rate = 0.2

    loss = 'mse'
    opti = 'adam'

    epochs = 5
    batch_size = 32

    # load training data
    train = np.array(pd.read_csv('train_train.csv', dtype={11:str, 12:str}))
    rows_to_train = min(train_max_rows, len(train))
    
    train_title = train[:rows_to_train, 8].astype(str)
    train_essay1 = train[:rows_to_train, 9].astype(str)
    train_essay2 = train[:rows_to_train, 10].astype(str)
    train_essay3 = train[:rows_to_train, 11].astype(str)
    train_essay4 = train[:rows_to_train, 12].astype(str)
    train_summary = train[:rows_to_train, 13].astype(str)

    train_approved = train[:rows_to_train, 15]

    # load valid data
    valid = np.array(pd.read_csv('train_valid.csv', dtype={11:str, 12:str}))
    rows_to_valid = min(valid_max_rows, len(valid))
    
    valid_title = valid[:rows_to_valid, 8].astype(str)
    valid_essay1 = valid[:rows_to_valid, 9].astype(str)
    valid_essay2 = valid[:rows_to_valid, 10].astype(str)
    valid_essay3 = valid[:rows_to_valid, 11].astype(str)
    valid_essay4 = valid[:rows_to_valid, 12].astype(str)
    valid_summary = valid[:rows_to_valid, 13].astype(str)

    valid_approved = valid[:rows_to_valid, 15:]

    print('rows to train : ' + str(rows_to_train))
    print('rows to valid : ' + str(rows_to_valid))

    # save validation output data
    RD.saveArray('bert_valid_rightAnswer.txt', valid_approved[:rows_to_valid], '\t', 500)

    # each text model for each dataset
    data_to_train = [train_title, train_essay1, train_essay2, train_essay3, train_essay4, train_summary]
    data_to_valid = [valid_title, valid_essay1, valid_essay2, valid_essay3, valid_essay4, valid_summary]
    text_models = []

    for i in range(6):
        text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab), embedding_dimensions=embedding_dim,
                            cnn_filters=cnn_filters, dnn_units=dnn_units, dropout_rate=dropout_rate)
        text_model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
        
        text_models.append(text_model)

    # train / valid result array
    train_result = [[0 for j in range(6)] for i in range(rows_to_train)]
    valid_result = [[0 for j in range(6)] for i in range(rows_to_valid)]

    # train / valid max length

    # for donorschoose-application-screening,
    # max_length_train = 132, 2183, 818, 387, 224, 234
    
    max_lengths = RD.loadArray('bert_max_lengths_train.txt')[0]

    print('max lengths:')
    print(max_lengths)

    # model 0: train_title   -> train_approved
    # model 1: train_essay1  -> train_approved
    # model 2: train_essay2  -> train_approved
    # model 3: train_essay3  -> train_approved
    # model 4: train_essay4  -> train_approved
    # model 5: train_summary -> train_approved
    for i in range(6):
        
        input_data = data_to_train[i]
        output_data = train_approved
        rows = len(input_data)

        # process dataset : convert into BERT-usable dataset
        processed_dataset = convertForBert(input_data, output_data, print_interval, tokenizer, int(max_lengths[i]))

        print('train')
        print(next(iter(processed_dataset)))

        # preparation for training
        batched_dataset = createBatchedDataset(processed_dataset, rows, batch_size)

        # TRAIN
        total_batches = math.ceil(rows / batch_size)
        train_data = batched_dataset.take(total_batches)
        text_models[i].fit(train_data, epochs=epochs)

        # update train result array
        for j in range(rows_to_train):
            train_result[j][i] = output_data[j]

    # validate using each model
    for i in range(6):

        input_data = data_to_valid[i]
        rows = len(input_data)

        # process dataset : convert into BERT-usable dataset
        valid_data = convertForBert(input_data, None, print_interval, tokenizer, int(max_lengths[i]))

        print('valid')
        print(next(iter(processed_dataset)))

        # VALIDATION
        prediction = text_models[i].predict(valid_data)

        print('prediction')
        print(np.shape(prediction))
        print(np.array(prediction))

        # update valid result array
        for j in range(rows_to_valid):
            valid_result[j][i] = prediction[j][0]

    # write result for training
    RD.saveArray('bert_train_result.txt', train_result, '\t', 500)

    # write result for validation
    RD.saveArray('bert_valid_result.txt', valid_result, '\t', 500)
