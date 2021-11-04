# ref: https://wikidocs.net/45101
#      https://huggingface.co/transformers/model_doc/roberta.html
#      https://www.kaggle.com/sourabhy/commonlit-roberta-ensemble-multichannel-cnn

import math
import numpy as np
import pandas as pd
import re
import main_helper as h
import NN_models as nns

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
    max_len = 0

    print('')
    
    for i in range(len(X)):
        roberta_input  = roberta_tokenizer(X[i], return_tensors="tf")
        roberta_output = roberta_model(roberta_input)
        roberta_lhs    = roberta_output.last_hidden_state # shape: (1, N, 768)
        roberta_lhs    = np.array(roberta_lhs)[0] # shape: (N, 768)

        encoded_X.append(roberta_lhs) # shape: (N, 768) x rows for different N's

        print(str(i) + ' / ' + str(len(X)) + ' : len=' + str(len(roberta_lhs)))
        max_len = max(max_len, len(roberta_lhs))

    print('')
        
    return (encoded_X, max_len)

if __name__ == '__main__':

    # read data
    (train_X, train_Y, test_X, ids) = h.loadData(100) # 999999

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
    print(np.shape(train_Y)) # (train rows)
    print(train_Y)
    
    # apply regular expression and tokenize input data
    train_X  = h.applyRegex(train_X)
    test_X   = h.applyRegex(test_X)

    pad_encoded_train_Xs = []
    pad_encoded_test_Xs  = []

    # encode input data (count) times with indices=0,1,...,(count-1) in 0..767
    (encoded_train_X, max_len_train) = encodeX(train_X, roberta_tokenizer, roberta_model)
    (encoded_test_X , max_len_test ) = encodeX(test_X , roberta_tokenizer, roberta_model)
    max_len = max(max_len_train, max_len_test)

    print('\n[04] encoded_train_X :')
    print('total:', np.shape(encoded_train_X))    # shape: (train rows)
    print('each :', np.shape(encoded_train_X[0])) # shape: (tokens, 768)
    print('first data:\n', np.array(encoded_train_X[0]))
                    
    print('\n[05] encoded_test_X :')
    print('total:', np.shape(encoded_test_X))    # shape: (test rows)
    print('each :', np.shape(encoded_test_X[0])) # shape: (tokens, 768)
    print('first data:\n', np.array(encoded_test_X[0]))
        
    # (tokens, 768) ---> (max_len, 768) for each row (fill with ZERO for remaining cells)
    encoded_train_Xs = []
    encoded_test_Xs  = []
        
    for i in range(trainLen):
        encoded_train_X_vecs = []
        
        for j in range(max_len):
            if j < len(encoded_train_X[i]):
                encoded_train_X_vecs.append(encoded_train_X[i][j])
            else:
                encoded_train_X_vecs.append([0 for k in range(768)])

        # append to dataset as flatten
        encoded_train_Xs.append(np.array(encoded_train_X_vecs).flatten())

    for i in range(testLen):
        
        encoded_test_X_vecs = []
        
        for j in range(max_len):
            if j < len(encoded_test_X[i]):
                encoded_test_X_vecs.append(encoded_test_X[i][j])
            else:
                encoded_test_X_vecs.append([0 for k in range(768)])

        # append to dataset as flatten
        encoded_test_Xs.append(np.array(encoded_test_X_vecs).flatten())

    # make as numpy array
    encoded_train_Xs = np.array(encoded_train_Xs)
    encoded_test_Xs  = np.array(encoded_test_Xs)

    print('\n[06] pad_encoded_train_X :')
    print(np.shape(encoded_train_Xs)) # (train rows, max_len * 768)
    print(encoded_train_Xs)
                    
    print('\n[07] pad_encoded_test_X :')
    print(np.shape(encoded_test_Xs)) # (test rows, max_len * 768)
    print(encoded_test_Xs)
        
    # multiply 3.0
    encoded_train_Xs = encoded_train_Xs * 3.0
    encoded_test_Xs  = encoded_test_Xs  * 3.0

    # find additional info
    encoded_train_Xs = h.getAdditionalInfo(encoded_train_Xs, trainLen, train_X)
    encoded_test_Xs  = h.getAdditionalInfo(encoded_test_Xs,  testLen,  test_X)

    print('\n[08] pad_encoded_train_X with additional info :')
    print(np.shape(encoded_train_Xs)) # (train rows, max_len * 768 + 3)
    print(encoded_train_Xs)

    print('\n[09] pad_encoded_test_X with additional info :')
    print(np.shape(encoded_test_Xs)) # (test rows, max_len * 768 + 3)
    print(encoded_test_Xs)

    # get model
    model = nns.TEXT_MODEL_LSTM0_ver01(None, 64, max_len)
    
    # train/valid using GPU
    valid_rate = 0.15
    h.trainOrValid(model, True, 'roberta', valid_rate, trainLen,
                   encoded_train_Xs, train_Y, encoded_test_Xs, avg, stddev)

    # test using GPU
    final_prediction = h.trainOrValid(model, False, 'roberta', valid_rate, trainLen,
                                      encoded_train_Xs, train_Y, encoded_test_Xs, avg, stddev)

    # final submission
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': np.array(final_prediction).flatten()
    })
    final_submission.to_csv('submission_roberta.csv', index=False)
