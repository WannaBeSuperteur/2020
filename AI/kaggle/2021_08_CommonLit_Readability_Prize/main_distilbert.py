# https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/

import math
import numpy as np
import pandas as pd
import re
import main_helper as h

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
    (train_X, train_Y, test_X, ids) = h.loadData(100)

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
    print(np.shape(train_Y))
    print(train_Y)
    
    # apply regular expression and tokenize input data
    train_X  = h.applyRegex(train_X)
    test_X   = h.applyRegex(test_X)
    max_lens = []

    pad_encoded_train_Xs = []
    pad_encoded_test_Xs  = []

    count = 4

    # encode input data 4 times with indices=0,1,...,19 in 0..767
    encoded_train_X = encodeX(train_X, distilbert_tokenizer, distilbert_model)
    encoded_test_X  = encodeX(test_X , distilbert_tokenizer, distilbert_model)

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
        
        pad_encoded_train_X = pad_encoded_train_X * 1.5
        pad_encoded_test_X  = pad_encoded_test_X  * 1.5
        
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
        pad_encoded_train_X = h.getAdditionalInfo(pad_encoded_train_X, trainLen, train_X, encoded_train_X)
        pad_encoded_test_X = h.getAdditionalInfo(pad_encoded_test_X, testLen, test_X, encoded_test_X)

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
        
    # train using GPU
    for i in range(count):
        try:
            # for local with GPU
            with tf.device('/gpu:0'):
                model = h.defineAndTrainModel(max_len, pad_encoded_train_X, train_Y)
                    
        except:
            # for kaggle notebook or local with no GPU
            print('cannot find GPU')
            model = h.defineAndTrainModel(max_len, pad_encoded_train_X, train_Y)
            
        # predict using model
        prediction = model.predict(pad_encoded_test_X).flatten()
        prediction = np.clip(prediction, 0.0001, 0.9999)

        print('\n[14] original (normalized and then SIGMOID-ed) prediction:')
        print(np.shape(prediction))
        print(prediction)

        # y -> denormalize(invSigmoid(y))
        for i in range(len(prediction)):
            prediction[i] = h.invSigmoid(prediction[i])
                
        prediction = prediction * stddev + avg

        # write final submission
        print('\n[15] converted prediction:')
        print(np.shape(prediction))
        print(prediction)

        try:
            final_prediction += prediction
        except:
            final_prediction = prediction

    # print final prediction
    final_prediction /= float(count)
    
    print('\n[16] FINAL prediction:')
    print(np.shape(final_prediction))
    print(final_prediction)

    # final submission
    final_submission = pd.DataFrame(
    {
        'id': ids,
        'target': final_prediction
    })
    final_submission.to_csv('submission_distilbert.csv', index=False)
