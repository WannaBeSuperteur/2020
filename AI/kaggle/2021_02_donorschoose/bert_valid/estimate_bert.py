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
import keras.backend.tensorflow_backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import bert

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

def tokenize_text(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self, vocabulary_size, precols,
                 embedding_dimensions=128, cnn_filters=50, dnn_units=512,
                 dropout_rate=0.1, training=False, name='text_model'):
        
        super(TEXT_MODEL, self).__init__(name=name)

        # to do: [V] split date into year, month, DOW and hour
        #        find the loss function for ROC-AUC (other than mean-squared-error)
        #        to find more valuable variables from codes
        #        [V] callbacks : val_loss (for early stopping) and lr_reduced
        #        [V] use the length of each text (title, essays, ...)
        #        [V] split project_subject_categories and project_subject_subcategories

        # i0_tp   : teacher_prefix                (   6 unique items)
        # i1_ss   : school_state                  (  51 unique items)
        # i2_d_y  : submit_date -> year           (   2 unique items)
        # i3_d_m  : submit_date -> month          (  12 unique items)
        # i4_d_d  : submit_date -> DOW            (   7 unique items)
        # i5_d_h  : submit_date -> hour           (  24 unique items)
        # i6_pgc  : project_grade_category        (   4 unique items)
        # i7_psc  : project_subject_categories    (   9 unique items)
        # i8_pss  : project_subject_subcategories (  30 unique items)
        # i9_len  :                                   6
        # i10_ts  :                                   1
        #
        # total   : i0_tp, i1_ss, ..., i10_ts     ( 152 unique columns)
        # text    : tokenized by BERT             ( 818        columns)
        #
        #                                         ( 970        columns)

        # constants
        self.u00, self.u01, self.u02, self.u03, self.u04 = 6, 51, 2, 12, 7
        self.u05, self.u06, self.u07, self.u08           = 24, 4, 9, 30
        
        self.e00, self.e01, self.e02, self.e03, self.e04 = 4, 12, 1, 4, 3,
        self.e05, self.e06, self.e07, self.e08           = 4, 4, 12, 20
        
        self.d00, self.d01, self.d02, self.d03, self.d04 = 12, 12, 12, 12, 12, # 4 -> 12
        self.d05, self.d06, self.d07, self.d08           = 12, 12, 12, 12
        
        self.d09, self.d09_final, self.d10, self.dL = 32, 12, 12, 12 # 4 -> 12

        # flatten layer
        self.flat = tf.keras.layers.Flatten()

        # for info 0, 1, 2, 3, 4, 5, 6, 7, 8 and i9_cont
        L2 = tf.keras.regularizers.l2(0.001)
        
        self.emb00 = tf.keras.layers.Embedding(self.u00, self.e00, trainable=True, name='emb00')
        self.emb01 = tf.keras.layers.Embedding(self.u01, self.e01, trainable=True, name='emb01')
        self.emb02 = tf.keras.layers.Embedding(self.u02, self.e02, trainable=True, name='emb02')
        self.emb03 = tf.keras.layers.Embedding(self.u03, self.e03, trainable=True, name='emb03')
        self.emb04 = tf.keras.layers.Embedding(self.u04, self.e04, trainable=True, name='emb04')
        self.emb05 = tf.keras.layers.Embedding(self.u05, self.e05, trainable=True, name='emb05')
        self.emb06 = tf.keras.layers.Embedding(self.u06, self.e06, trainable=True, name='emb06')
        self.emb07 = tf.keras.layers.Embedding(self.u07, self.e07, trainable=True, name='emb07')
        self.emb08 = tf.keras.layers.Embedding(self.u08, self.e08, trainable=True, name='emb08')
        
        self.den00 = tf.keras.layers.Dense(units=self.d00, activation='relu', kernel_regularizer=L2, name='den00')
        self.den01 = tf.keras.layers.Dense(units=self.d01, activation='relu', kernel_regularizer=L2, name='den01')
        self.den02 = tf.keras.layers.Dense(units=self.d02, activation='relu', kernel_regularizer=L2, name='den02')
        self.den03 = tf.keras.layers.Dense(units=self.d03, activation='relu', kernel_regularizer=L2, name='den03')
        self.den04 = tf.keras.layers.Dense(units=self.d04, activation='relu', kernel_regularizer=L2, name='den04')
        self.den05 = tf.keras.layers.Dense(units=self.d05, activation='relu', kernel_regularizer=L2, name='den05')
        self.den06 = tf.keras.layers.Dense(units=self.d06, activation='relu', kernel_regularizer=L2, name='den06')
        self.den07 = tf.keras.layers.Dense(units=self.d07, activation='relu', kernel_regularizer=L2, name='den07')
        self.den08 = tf.keras.layers.Dense(units=self.d08, activation='relu', kernel_regularizer=L2, name='den08')
        
        self.den09_0 = tf.keras.layers.Dense(units=self.d09, activation='relu', kernel_regularizer=L2, name='CONT_den09_0')
        self.den09_1 = tf.keras.layers.Dense(units=self.d09, activation='relu', kernel_regularizer=L2, name='CONT_den09_1')
        self.den09_2 = tf.keras.layers.Dense(units=self.d09, activation='relu', kernel_regularizer=L2, name='CONT_den09_2')
        self.den09_final = tf.keras.layers.Dense(units=self.d09_final, activation='relu', kernel_regularizer=L2, name='CONT_den09_final')
        self.den10 = tf.keras.layers.Dense(units=self.d10, activation='relu', kernel_regularizer=L2, name='CONT_den10')

        # for BERT-tokenized text
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions, trainable=True, name='BERT_embedding')
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding='valid', activation='relu', name='BERT_cnn1')
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu', name='BERT_cnn2')
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu', name='BERT_cnn3')
        self.pool = tf.keras.layers.GlobalMaxPool1D(name='BERT_pooling')
        self.denL = tf.keras.layers.Dense(units=self.dL, activation='relu', kernel_regularizer=L2, name='BERT_dense')

        # for concatenated layer and output
        self.dense = tf.keras.layers.Dense(units=dnn_units, activation='relu', kernel_regularizer=L2, name='FINAL_dense0')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.last_dense = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output') # output layer

    def call(self, inputs, training):

        # split input
        i0_tp, i1_ss, i2_d_y, i3_d_m, i4_d_d, i5_d_h, i6_pgc, i7_psc, i8_pss, i9_len, i10_ts, inputs_text =\
        tf.split(inputs, [self.u00, self.u01, self.u02, self.u03, self.u04, self.u05, self.u06, self.u07, self.u08, 6, 1, 818], 1)

        inputs_text = tf.cast(inputs_text, tf.int32)

        # info 0, 1, 2, 3, 4, 5, 6, 7 and 8
        i0_emb = self.emb00(i0_tp)
        i0_emb = self.flat(i0_emb)
        i0_den = self.den00(i0_emb)
        i0_drop = self.dropout(i0_den, training)
        
        i1_emb = self.emb01(i1_ss)
        i1_emb = self.flat(i1_emb)
        i1_den = self.den01(i1_emb)
        i1_drop = self.dropout(i1_den, training)
        
        i2_emb = self.emb02(i2_d_y)
        i2_emb = self.flat(i2_emb)
        i2_den = self.den02(i2_emb)
        i2_drop = self.dropout(i2_den, training)
        
        i3_emb = self.emb03(i3_d_m)
        i3_emb = self.flat(i3_emb)
        i3_den = self.den03(i3_emb)
        i3_drop = self.dropout(i3_den, training)
        
        i4_emb = self.emb04(i4_d_d)
        i4_emb = self.flat(i4_emb)
        i4_den = self.den04(i4_emb)
        i4_drop = self.dropout(i4_den, training)

        i5_emb = self.emb05(i5_d_h)
        i5_emb = self.flat(i5_emb)
        i5_den = self.den05(i5_emb)
        i5_drop = self.dropout(i5_den, training)
        
        i6_emb = self.emb06(i6_pgc)
        i6_emb = self.flat(i6_emb)
        i6_den = self.den06(i6_emb)
        i6_drop = self.dropout(i6_den, training)
        
        i7_emb = self.emb07(i7_psc)
        i7_emb = self.flat(i7_emb)
        i7_den = self.den07(i7_emb)
        i7_drop = self.dropout(i7_den, training)
        
        i8_emb = self.emb08(i8_pss)
        i8_emb = self.flat(i8_emb)
        i8_den = self.den08(i8_emb)
        i8_drop = self.dropout(i8_den, training)
        
        i9_den0 = self.den09_0(i9_len)
        i9_den1 = self.den09_1(i9_den0)
        i9_den2 = self.den09_2(i9_den1)
        i9_denFinal = self.den09_final(i9_den2)
        i9_drop = self.dropout(i9_denFinal, training)

        i10_den = self.den10(i10_ts)
        i10_drop = self.dropout(i10_den, training)

        # BERT-tokenized text
        l = self.embedding(inputs_text)
        
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)
        
        l_concat = tf.concat([l_1, l_2, l_3], axis=-1)
        l_den = self.denL(l_concat)
        l_drop = self.dropout(l_den, training)

        # concatenate embedded info and concatenated BERT-tokenized text
        concatenated = tf.concat([i0_drop, i1_drop, i2_drop, i3_drop, i4_drop, i5_drop,
                                  i6_drop, i7_drop, i8_drop, i9_drop, i10_drop, l_drop], axis=-1)
        
        concatenated = self.dense(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output

def convertForBert(input_data, print_interval, tokenizer, max_length, precols):

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

        return np.array(tokenized_input)

    # for valid
    except:
        return np.array(tokenized_input)

def createBatchedDataset(processed_dataset, rows, batch_size):
    batched_dataset = processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))
        
    total_batches = math.ceil(rows / batch_size)
    batched_dataset.shuffle(total_batches)

    return batched_dataset

def mainFunc(count, tokenizer):

    # define configuration
    train_max_rows = 9999999
    valid_max_rows = 9999999
    print_interval = 400
    batch_size = 32

    embedding_dim = 200
    cnn_filters = 50
    dnn_units = 256
    dropout_rate = 0.5

    loss = 'mse'
    opti = optimizers.Adam(0.0005, decay=1e-6)

    epochs = 15
    batch_size = 32

    trainFile = 'train.csv'
    validFile = 'test.csv'
    trainExtractedFile = 'train_extracted.txt'
    validExtractedFile = 'test_extracted.txt'
    validAnswerFile = None
    isValidTest = True # test mode

    # load training data
    train = np.array(pd.read_csv(trainFile, dtype={11:str, 12:str}))
    rows_to_train = min(train_max_rows, len(train))
    
    train_title = train[:rows_to_train, 8].astype(str)
    train_essay1 = train[:rows_to_train, 9].astype(str)
    train_essay2 = train[:rows_to_train, 10].astype(str)
    train_essay3 = train[:rows_to_train, 11].astype(str)
    train_essay4 = train[:rows_to_train, 12].astype(str)
    train_summary = train[:rows_to_train, 13].astype(str)

    train_approved = train[:rows_to_train, 15].astype(float)

    # load valid data
    valid = np.array(pd.read_csv(validFile, dtype={11:str, 12:str}))
    rows_to_valid = min(valid_max_rows, len(valid))
    
    valid_title = valid[:rows_to_valid, 8].astype(str)
    valid_essay1 = valid[:rows_to_valid, 9].astype(str)
    valid_essay2 = valid[:rows_to_valid, 10].astype(str)
    valid_essay3 = valid[:rows_to_valid, 11].astype(str)
    valid_essay4 = valid[:rows_to_valid, 12].astype(str)
    valid_summary = valid[:rows_to_valid, 13].astype(str)

    valid_approved = valid[:rows_to_valid, 15:].astype(float)

    # print the number of rows to train/valid
    print('\n[00] rows to train : ' + str(rows_to_train))
    print('\n[01] rows to valid : ' + str(rows_to_valid))

    # extract data from training dataset
    train_info = []
    valid_info = []

    option = [-1, -1, 2, 2, 7, 2, 8, 8, 5, 5, 5, 5, 5, 5, 1, -1]
    title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    wordCount = ME.getWordCount(option, trainFile)

    try:
        train_extracted = RD.loadArray(trainExtractedFile, '\t')
        valid_extracted = RD.loadArray(validExtractedFile, '\t')
        
    except:
        (train_extracted, _, onehot) = ME.extract(trainFile, option, title, wordCount, None, [2016, 2017])
        (valid_extracted, _, _) = ME.extract(validFile, option, title, wordCount, onehot, [2016, 2017])

        RD.saveArray(trainExtractedFile, train_extracted, '\t', 500)
        RD.saveArray(validExtractedFile, valid_extracted, '\t', 500)

    precols = len(train_extracted[0])

    print('\n[02] precols:')
    print(precols)

    for i in range(rows_to_train):
        train_info.append(train_extracted[i])

    for i in range(rows_to_valid):
        valid_info.append(valid_extracted[i])

    print('\n[03] train info:')
    print(np.shape(train_info))
    print(np.array(train_info))

    print('\n[04] valid info:')
    print(np.shape(valid_info))
    print(np.array(valid_info))

    # save validation output data (only for VALID mode)
    if isValidTest == False:
        try:
            valid_rightAnswer = RD.loadArray(validAnswerFile)
        except:
            RD.saveArray(validAnswerFile, valid_approved[:rows_to_valid], '\t', 500)

    with K.tf.device('/gpu:0'):
        
        # each text model for each dataset
        text_to_train = [train_title, train_essay1, train_essay2, train_essay3, train_essay4, train_summary]
        text_to_valid = [valid_title, valid_essay1, valid_essay2, valid_essay3, valid_essay4, valid_summary]
        text_models = []

        for i in range(6):
            text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab), precols=precols,
                                    embedding_dimensions=embedding_dim,
                                    cnn_filters=cnn_filters,
                                    dnn_units=dnn_units, dropout_rate=dropout_rate)
            
            text_model.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
            
            text_models.append(text_model)
            
        # train / valid result array
        train_result = [[0 for j in range(6)] for i in range(rows_to_train)]
        valid_result = [[0 for j in range(6)] for i in range(rows_to_valid)]

        # train / valid max length

        # for donorschoose-application-screening,
        # max_length_train = 132, 2183, 818, 387, 224, 234
        
        max_lengths = RD.loadArray('bert_max_lengths_train.txt')[0]

        print('\n[05] max lengths:')
        print(max_lengths)

        # model 0: train_title   -> train_approved
        # model 1: train_essay1  -> train_approved
        # model 2: train_essay2  -> train_approved
        # model 3: train_essay3  -> train_approved
        # model 4: train_essay4  -> train_approved
        # model 5: train_summary -> train_approved
        for i in range(2, 3): # 6
            
            rows = len(text_to_train[i])

            # process dataset : convert into BERT-usable dataset
            tokenized_input = convertForBert(text_to_train[i], print_interval, tokenizer, int(max_lengths[i]), precols)
            
            train_text = np.array(tokenized_input).astype(float)
            train_info = np.array(train_info).astype(float)

            print('\n[06] train text')
            print(np.shape(train_text))
            print('------')
            print(np.array(train_text))

            print('\n[07] train info')
            print(np.shape(train_info))
            print('------')
            print(np.array(train_info))

            train_data = np.concatenate((train_info, train_text), axis=1).astype(float)

            print('\n[08] train info+text (final input)')
            print(np.shape(train_data))
            print('------')
            print(np.array(train_data))

            print('\n[09] train approved (final output)')
            print(np.shape(train_approved))
            print('------')
            print(np.array(train_approved[:100]))

            # callback list for training
            early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3)
            lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=0.0001, mode='min')

            ### TRAIN THE MODEL ###
            text_models[i].fit(train_data, train_approved, validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)
            text_models[i].summary()

            # update train result array
            for j in range(rows_to_train):
                train_result[j][i] = train_approved[j]

            # save the model
            text_models[i].save('model_' + str(i) + '_count_' + str(count) + '_train_' + str(train_max_rows) + '_e_' + str(epochs))

        # validate using each model
        for i in range(2, 3): # 6

            rows = len(text_to_valid[i])

            # process dataset : convert into BERT-usable dataset
            valid_text = convertForBert(text_to_valid[i], print_interval, tokenizer, int(max_lengths[i]), precols)
            
            valid_text = np.array(valid_text).astype(float)
            valid_info = np.array(valid_info).astype(float)

            print('\n[10] valid text')
            print(np.shape(valid_text))
            print('------')
            print(np.array(valid_text))

            print('\n[11] valid info')
            print(np.shape(valid_info))
            print('------')
            print(np.array(valid_info))

            valid_data = np.concatenate((valid_info, valid_text), axis=1).astype(float)

            print('\n[12] valid info+text')
            print(np.shape(valid_data))
            print('------')
            print(np.array(valid_data))

            # load the model
            loaded_model = tf.keras.models.load_model('model_' + str(i) + '_count_' + str(count) + '_train_' + str(train_max_rows) + '_e_' + str(epochs))

            # VALIDATION
            prediction = loaded_model.predict(valid_data)

            print('\n[13] prediction')
            print(np.shape(prediction))
            print(np.array(prediction))

            # update valid result array
            for j in range(rows_to_valid):
                valid_result[j][i] = prediction[j][0]

    # write result for training
    RD.saveArray('bert_train_result_count_' + str(count) + '.txt', train_result, '\t', 500)

    # write result for validation
    RD.saveArray('bert_valid_result_count_' + str(count) + '.txt', valid_result, '\t', 500)

if __name__ == '__main__':

    # create bert tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # repeat 4 times
    times = 4
    
    for i in range(times):
        mainFunc(i, tokenizer)
