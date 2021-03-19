# train_train.csv, train_valid.csv -> bert_train_result.txt, bert_valid_result.txt
# additional need: bert_max_lengths_train.txt

# ref: https://insightcampus.co.kr/insightcommunity/?mod=document&uid=12854
#      https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

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

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import bert

def tokenize_text(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self, vocabulary_size, precols,
                 embedding_dimensions=128, cnn_filters=50, dnn_units=512,
                 dropout_rate=0.1, training=False, name='text_model'):
        
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

        #print('============')
        #print(inputs)
        #print('============')

        # 864 is temp
        inputs_pr, inputs_text = tf.split(inputs, [precols, 864 - precols], 1)
        inputs_text = tf.cast(inputs_text, tf.int32)

        #print('============')
        #print(inputs_pr)
        #print('============')
        #print(inputs_text)
        #print('============')
        
        l = self.embedding(inputs_text)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)
        
        concatenated = tf.concat([inputs_pr, l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output

def convertForBert(input_data, output_data, print_interval, tokenizer, max_length, precols):

    rows = len(input_data)
    
    # convert training/valid input data into the list of word IDs
    tokenized_input = []

    new_max_length = 0
        
    for i in range(rows):
        if i % print_interval == 0: print(i)

        if i == 0:
            print('\n[c00] text to tokenize:')
            print(input_data[i][precols])
        
        tokenized_txt = tokenize_text(input_data[i][precols], tokenizer)

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

        print('\n[c02] shape of input data:')
        print(np.shape(input_data))

        # bind input and output data
        # in_out: set of [[info + text_input], output]
        #         info       = input_data[i][:precols]
        #         text_input = tokenized_input[i]
        in_out = []
        for i in range(rows):
            in_out.append([np.concatenate([input_data[i][:precols], tokenized_input[i]]),
                           output_data[i]])

        print('\n[c03] shape of in_out:')
        print('set of [[info + text_input], output] :\n' + str(np.shape(in_out)))
        print('each   [[info + text_input], output] :\n' + str(np.shape(in_out[0])))
        print('each   [info + text_input]           :\n' + str(np.shape(in_out[0][0])))
        
        in_out_labels = [(row[0], row[1]) for row in in_out]
        
        processed_dataset = tf.data.Dataset.from_generator(lambda: in_out_labels, output_types=(tf.float32, tf.float32))
        return processed_dataset

    # for valid
    except:
        print('\n[c04] shape of tokenized input:')
        print(np.shape(tokenized_input))
        
        print('\n[c05] shape of input data:')
        print(np.shape(input_data))

        print('\n[c06] concatenation:')
        print(np.shape(np.concatenate((input_data[:, :precols], tokenized_input), axis=1)))
        
        return np.concatenate((input_data[:, :precols], tokenized_input), axis=1)

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
    train_max_rows = 1000 # 9999999 (no limit)
    valid_max_rows = 1000 # 9999999 (no limit)
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

    # print the number of rows to train/valid
    print('\n[00] rows to train : ' + str(rows_to_train))
    print('\n[01] rows to valid : ' + str(rows_to_valid))

    # extract data from training dataset
    train_info = []
    valid_info = []

    option = [-1, -1, 2, -1, -1, 2, 3, 3, -1, -1, -1, -1, -1, -1, 1, -1]
    title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    (train_extracted, _) = ME.extract('train_train.csv', option, title, -1)
    (valid_extracted, _) = ME.extract('train_valid.csv', option, title, -1)
    precols = len(train_extracted[0])

    print('\n[02] precols:')
    print(precols)

    for i in range(rows_to_train):
        train_info.append(train_extracted[i])

    for i in range(rows_to_train):
        valid_info.append(valid_extracted[i])

    print('\n[03] train info:')
    print(np.shape(train_info))
    print(np.array(train_info))

    print('\n[04] valid info:')
    print(np.shape(valid_info))
    print(np.array(valid_info))

    # save validation output data
    RD.saveArray('bert_valid_rightAnswer.txt', valid_approved[:rows_to_valid], '\t', 500)

    # each text model for each dataset
    data_to_train = [train_title, train_essay1, train_essay2, train_essay3, train_essay4, train_summary]
    data_to_valid = [valid_title, valid_essay1, valid_essay2, valid_essay3, valid_essay4, valid_summary]
    text_models = []

    for i in range(6):
        text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab), precols=precols,
                                embedding_dimensions=embedding_dim,
                                cnn_filters=cnn_filters,
                                dnn_units=dnn_units, dropout_rate=dropout_rate)
        
        text_model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
        
        text_models.append(text_model)

    # concatenate train_info with data_to_train
    # concatenate valid_info with data_to_valid
    for i in range(6):

        print('\n[05/' + str(i) + '] data to train:')
        print(np.shape(data_to_train[i]))
        
        print('\n[06/' + str(i) + '] data to valid:')
        print(np.shape(data_to_valid[i]))
        
        #    data_to_train/valid[i] = ['a', 'b', 'c', ...]
        # -> data_to_train/valid_   = [['a'], ['b'], ['c'], ...] 
        data_to_train_ = []
        data_to_valid_ = []
        
        for j in range(rows_to_train):
            data_to_train_.append([data_to_train[i][j]])

        for j in range(rows_to_valid):
            data_to_valid_.append([data_to_valid[i][j]])

        data_to_train[i] = np.concatenate((train_info, data_to_train_), axis=1)
        data_to_valid[i] = np.concatenate((valid_info, data_to_valid_), axis=1)

        print('\n[07/' + str(i) + '] data to train (concatenated):')
        print(np.shape(data_to_train[i]))
        
        print('\n[08/' + str(i) + '] data to valid (concatenated):')
        print(np.shape(data_to_valid[i]))
        
    # train / valid result array
    train_result = [[0 for j in range(6)] for i in range(rows_to_train)]
    valid_result = [[0 for j in range(6)] for i in range(rows_to_valid)]

    # train / valid max length

    # for donorschoose-application-screening,
    # max_length_train = 132, 2183, 818, 387, 224, 234
    
    max_lengths = RD.loadArray('bert_max_lengths_train.txt')[0]

    print('\n[09] max lengths:')
    print(max_lengths)

    # model 0: train_title   -> train_approved
    # model 1: train_essay1  -> train_approved
    # model 2: train_essay2  -> train_approved
    # model 3: train_essay3  -> train_approved
    # model 4: train_essay4  -> train_approved
    # model 5: train_summary -> train_approved
    for i in range(2, 3): # 6
        
        input_data = data_to_train[i]
        output_data = train_approved
        rows = len(input_data)

        # process dataset : convert into BERT-usable dataset
        processed_dataset = convertForBert(input_data, output_data,
                                           print_interval, tokenizer, int(max_lengths[i]), precols)

        print('\n[10] train')
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

        # save the model
        text_models[i].save('model_' + str(i) + '_train_' + str(train_max_rows) + '_e_' + str(epochs))

    # validate using each model
    for i in range(2, 3): # 6

        input_data = data_to_valid[i]
        rows = len(input_data)

        # process dataset : convert into BERT-usable dataset
        valid_data = convertForBert(input_data, None,
                                    print_interval, tokenizer, int(max_lengths[i]), precols)
        valid_data = np.array(valid_data).astype(float)

        print('\n[11] valid')
        print(np.shape(valid_data))
        print(np.array(valid_data))

        # load the model
        loaded_model = tf.keras.models.load_model('model_' + str(i) + '_train_' + str(train_max_rows) + '_e_' + str(epochs))

        # VALIDATION
        prediction = loaded_model.predict(valid_data)

        print('\n[12] prediction')
        print(np.shape(prediction))
        print(np.array(prediction))

        # update valid result array
        for j in range(rows_to_valid):
            valid_result[j][i] = prediction[j][0]

    # write result for training
    RD.saveArray('bert_train_result.txt', train_result, '\t', 500)

    # write result for validation
    RD.saveArray('bert_valid_result.txt', valid_result, '\t', 500)
