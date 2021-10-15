# ref: https://wikidocs.net/45101
#      https://huggingface.co/transformers/model_doc/roberta.html

import math
import numpy as np
import pandas as pd
import re
import main_helper as h

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

np.set_printoptions(linewidth=180, edgeitems=5)

# encode and add padding to the text
def encodeX(X, roberta_tokenizer, roberta_model):
    encoded_X = []

    print('')
    
    for i in range(len(X)):
        roberta_input  = roberta_tokenizer(X[i], return_tensors="tf")
        roberta_output = roberta_model(roberta_input)
        roberta_lhs    = roberta_output.last_hidden_state # shape: (1, N, 768)
        roberta_lhs    = np.array(roberta_lhs)[0] # shape: (N, 768)

        encoded_X.append(roberta_lhs) # shape: (N, 768) x rows for different N's

        print(str(i) + ' / ' + str(len(X)) + ' : len=' + str(len(roberta_lhs)))

    print('')
        
    return encoded_X

if __name__ == '__main__':

    # read data
    (train_X, train_Y, test_X, ids) = h.loadData(999999)

    trainLen = len(train_X)
    testLen = len(test_X)

    print('\n[00] original train_Y :')
    print(np.shape(train_Y))
    print(train_Y)

    # roBERTa tokenizer and model
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    
    # normalize output data : y -> sigmoid(normalize(y))
    (train_Y, avg, stddev) = h.normalize(train_Y)
    print('\n[01] avg=' + str(round(avg, 6)))
    print('\n[02] stddev=' + str(round(stddev, 6)))

    print('\n[03] converted train_Y :')
    print(np.shape(train_Y))
    print(train_Y)
    
    # apply regular expression and tokenize input data
    train_X  = h.applyRegex(train_X)
    test_X   = h.applyRegex(test_X)
    max_lens = []

    pad_encoded_train_Xs = []
    pad_encoded_test_Xs  = []

    count = 20

    # encode input data (count) times with indices=0,1,...,(count-1) in 0..767
    encoded_train_X = encodeX(train_X, roberta_tokenizer, roberta_model)
    encoded_test_X  = encodeX(test_X , roberta_tokenizer, roberta_model)

    print('\n[04] encoded_train_X :')
    print(np.shape(encoded_train_X))
            
    print('\n[05] encoded_test_X :')
    print(np.shape(encoded_test_X))
        
    for i in range(count):
        print(str(i) + ' / ' + str(count))

        # extract i-th index (i = 0..767) from encoded data
        # (N, 768) x rows for different N's ---> (N) x rows for different N's
        encoded_train_X_i = []
        encoded_test_X_i  = []
        
        for j in range(trainLen):
            encoded_train_X_ij = []
            for k in range(len(encoded_train_X[j])): encoded_train_X_ij.append(encoded_train_X[j][k][i])
            encoded_train_X_i.append(encoded_train_X_ij)

        for j in range(testLen):
            encoded_test_X_ij = []
            for k in range(len(encoded_test_X[j])): encoded_test_X_ij.append(encoded_test_X[j][k][i])
            encoded_test_X_i.append(encoded_test_X_ij)

        if i == 0:
            print('\n[06] encoded_train_X for i==0 :')
            print(np.shape(encoded_train_X_i))

            print('\n[07] encoded_test_X for i==0 :')
            print(np.shape(encoded_test_X_i))
        
        # add padding to input data
        (pad_encoded_train_X, pad_encoded_test_X, max_len) = h.addPadding(encoded_train_X_i, encoded_test_X_i)

        pad_encoded_train_X = pad_encoded_train_X * 3.0
        pad_encoded_test_X  = pad_encoded_test_X  * 3.0

        max_lens.append(max_len)

        if i == 0:
            print('\n[08] max_len=' + str(max_len))

        if i < 10:
            print('\n[09] pad_encoded_train_X (' + str(i) + ') :')
            print(np.shape(pad_encoded_train_X))
            print(pad_encoded_train_X)
            
            print('\n[10] pad_encoded_test_X  (' + str(i) + ') :')
            print(np.shape(pad_encoded_test_X))
            print(pad_encoded_test_X)
        
        # find additional info
        pad_encoded_train_X = h.getAdditionalInfo(pad_encoded_train_X, trainLen, train_X, encoded_train_X_i)
        pad_encoded_test_X = h.getAdditionalInfo(pad_encoded_test_X, testLen, test_X, encoded_test_X_i)

        pad_encoded_train_Xs.append(pad_encoded_train_X)
        pad_encoded_test_Xs.append(pad_encoded_test_X)

        if i == 0:
            print('\n[11] pad_encoded_train_X with additional info :')
            print(np.shape(pad_encoded_train_X))
            print(pad_encoded_train_X)

            print('\n[12] pad_encoded_test_X with additional info :')
            print(np.shape(pad_encoded_test_X))
            print(pad_encoded_test_X)

    print('\n[13] max_len\'s :')
    print(max_lens)
    
    # train/valid using GPU
    valid_rate = 0.15
    h.trainOrValid(True, 'roberta', valid_rate, trainLen, max_lens, pad_encoded_train_Xs, train_Y, pad_encoded_test_Xs,
                   count, avg, stddev)

    # test using GPU
    final_prediction = h.trainOrValid(False, 'roberta', valid_rate, trainLen, max_lens, pad_encoded_train_Xs, train_Y, pad_encoded_test_Xs,
                                      count, avg, stddev)

    # final submission
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': final_prediction
    })
    final_submission.to_csv('submission_roberta.csv', index=False)
