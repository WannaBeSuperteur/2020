# ref: https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/
#      https://www.kaggle.com/sourabhy/commonlit-roberta-ensemble-multichannel-cnn

import math
import numpy as np
import pandas as pd
import re
import main_helper as h
import NN_models as nns

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

np.set_printoptions(linewidth=180, edgeitems=5)

# encode and add padding to the text
def encodeX(X, distilbert_tokenizer, distilbert_model):
    encoded_X = []

    print('')
    
    for i in range(len(X)):
        distilbert_input  = distilbert_tokenizer(X[i], return_tensors="pt")
        distilbert_output = distilbert_model(**distilbert_input)
        distilbert_out    = distilbert_output[0].detach().numpy()[0]

        encoded_X.append(distilbert_out) # shape: (N, 768) x rows for different N's

        print(str(i) + ' / ' + str(len(X)) + ' : len=' + str(len(distilbert_out)))

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

    # DistilBert tokenizer and model
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model     = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
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

    useAtOnce = 10

    # encode input data (count) times with indices=0,1,...,(count-1) in 0..767
    encoded_train_X = encodeX(train_X, distilbert_tokenizer, distilbert_model)
    encoded_test_X  = encodeX(test_X , distilbert_tokenizer, distilbert_model)

    print('\n[04] encoded_train_X :')
    print('total:', np.shape(encoded_train_X)) # (train rows)
    print('each :', np.shape(encoded_train_X[0])) # (tokens, 768)
    print('first data:\n', np.array(encoded_train_X[0]))
                    
    print('\n[05] encoded_test_X :')
    print('total:', np.shape(encoded_test_X)) # (test rows)
    print('each :', np.shape(encoded_test_X[0])) # (tokens, 768)
    print('first data:\n', np.array(encoded_test_X[0]))
            
    # extract from 1st index to (useAtOnce)-th index (in 0..767) from encoded data
    # (tokens, 768) for each row ---> (tokens, useAtOnce) for each row
    encoded_train_Xs = []
    encoded_test_Xs  = []
        
    for i in range(trainLen):

        # flatten of [encoded_X[0], encoded_X[1], ..., encoded_X[useAtOnce-1]] (size: useAtOnce * 768)
        encoded_train_X_vecs = []
        for j in range(useAtOnce): encoded_train_X_vecs.append(encoded_train_X[i][j])
        encoded_train_X_vecs = np.array(encoded_train_X_vecs).flatten()
        
        encoded_train_Xs.append(encoded_train_X_vecs)

    for i in range(testLen):
        
        # flatten of [encoded_X[0], encoded_X[1], ..., encoded_X[useAtOnce-1]] (size: useAtOnce * 768)
        encoded_test_X_vecs = []
        for j in range(useAtOnce): encoded_test_X_vecs.append(encoded_test_X[i][j])
        encoded_test_X_vecs = np.array(encoded_test_X_vecs).flatten()
        
        encoded_test_Xs.append(encoded_test_X_vecs)

    encoded_train_Xs = np.array(encoded_train_Xs)
    encoded_test_Xs  = np.array(encoded_test_Xs)

    print('\n[06] pad_encoded_train_X :')
    print(np.shape(encoded_train_Xs)) # (train rows, useAtOnce * 768)
    print(encoded_train_Xs)
                    
    print('\n[07] pad_encoded_test_X :')
    print(np.shape(encoded_test_Xs)) # (test rows, useAtOnce * 768)
    print(encoded_test_Xs)
            
    # multiply 1.5
    encoded_train_Xs = encoded_train_Xs * 1.5
    encoded_test_Xs  = encoded_test_Xs  * 1.5
            
    # find additional info
    encoded_train_Xs = h.getAdditionalInfo(encoded_train_Xs, trainLen, train_X)
    encoded_test_Xs  = h.getAdditionalInfo(encoded_test_Xs,  testLen,  test_X)

    print('\n[08] pad_encoded_train_X with x1.5 and additional info :')
    print(np.shape(encoded_train_Xs)) # (train rows, useAtOnce * 768 + 3)
    print(encoded_train_Xs)
                    
    print('\n[09] pad_encoded_test_X with x1.5 and additional info :')
    print(np.shape(encoded_test_Xs)) # (test rows, useAtOnce * 768 + 3)
    print(encoded_test_Xs)

    # get model
    model = nns.TEXT_MODEL_LSTM0(useAtOnce, None, 64)
        
    # train/valid using GPU
    valid_rate = 0.15
    h.trainOrValid(model, True, 'distilbert', valid_rate, trainLen, useAtOnce,
                   encoded_train_Xs, train_Y, encoded_test_Xs, avg, stddev)

    # test using GPU
    final_prediction = h.trainOrValid(model, False, 'distilbert', valid_rate, trainLen, useAtOnce,
                                      encoded_train_Xs, train_Y, encoded_test_Xs, avg, stddev)

    # final submission
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': np.array(final_prediction).flatten()
    })
    final_submission.to_csv('submission_distilbert.csv', index=False)
