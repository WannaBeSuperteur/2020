import math
import numpy as np
import pandas as pd
import re
import tensorflow as tf

# load the dataset
def loadData(limit):
    try:
        # for local
        trainData = pd.read_csv('train.csv')
        testData = pd.read_csv('test.csv')
        
    except:
        # for kaggle notebook
        print('now running kaggle notebook')
        trainData = pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
        testData = pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')

    train_X = np.array(trainData['excerpt'])[:min(len(trainData), limit)]
    train_Y = np.array(trainData['target'])[:min(len(trainData), limit)]
    test_X = np.array(testData['excerpt'])

    return (train_X, train_Y, test_X, testData['id'])

# apply regular expression to the text
def applyRegex(all_X):
    regex = re.compile('[^a-zA-Z0-9.,?! ]')

    # remove excpet for a-z, A-Z, 0-9, '.', ',' and space
    for i in range(len(all_X)):
        all_X[i] = regex.sub('', str(all_X[i]))

    return all_X

"""
def addPadding(encoded_train_X, encoded_test_X):
    max_len_train = max(len(x) for x in encoded_train_X)
    max_len_test = max(len(x) for x in encoded_test_X)
    max_len = max(max_len_train, max_len_test)

    # add padding
    new_encoded_train_X = np.zeros((len(encoded_train_X), max_len))
    new_encoded_test_X  = np.zeros((len(encoded_test_X) , max_len))

    for i in range(len(new_encoded_train_X)):
        new_encoded_train_X[i][max_len - len(encoded_train_X[i]):] = encoded_train_X[i]
    for i in range(len(new_encoded_test_X)):
        new_encoded_test_X [i][max_len - len(encoded_test_X[i]) :] = encoded_test_X [i]

    return (new_encoded_train_X, new_encoded_test_X, max_len)
"""

# train the model
def trainModel(model, X, Y, validation_split, epochs, early_patience, lr_reduced_factor, lr_reduced_patience):

    early = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                             mode="min",
                                             patience=early_patience)
    
    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=lr_reduced_factor,
                                                      patience=lr_reduced_patience,
                                                      verbose=1,
                                                      min_delta=0.0001,
                                                      mode='min')
        
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, validation_split=validation_split, callbacks=[early, lr_reduced], epochs=epochs)
    model.summary()

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

# find info : number of sentences, average sentence length (in words), and total length (in words)
def getSentenceInfo(sentence):

    numberOfSentences = len(sentence.split('.'))
    totalLength       = len(sentence.split(' '))
    avgLength         = totalLength / numberOfSentences

    # return the info
    return np.array([numberOfSentences, totalLength, avgLength])

# find additional info
def getAdditionalInfo(pad_encoded_X, leng, X):

    additionalInfo = np.zeros((leng, 3))

    for i in range(leng):

        # add { number of sentences, average sentence length (in words), total length (in words) }
        # to encoded train_X
        additionalInfo[i] = getSentenceInfo(X[i])

    # normalize additional info
    for col in range(3):
        (additionalInfo[:, col], avg, stddev) = normalize(additionalInfo[:, col])
    
    pad_encoded_X = np.concatenate((pad_encoded_X, additionalInfo), axis=1)

    return pad_encoded_X

# train model using GPU
def trainOrValid(model, valid, algo_name, valid_rate, trainLen, useAtOnce, train_Xs, train_Y, test_Xs, avg, stddev):

    if valid == False: valid_rate = 0.0 # test
    trainCount = int((1.0 - valid_rate) * trainLen)

    # save original validation data
    if valid == True:
        train_Y_reshaped = np.reshape(train_Y[trainCount:].copy(), (trainLen - trainCount, 1))
        for i in range(len(train_Y_reshaped)): train_Y_reshaped[i] = invSigmoid(train_Y_reshaped[i])
        pd.DataFrame(train_Y_reshaped).to_csv('valid_original.csv')

    # train model
    epochs              = 100
    early_patience      = 6
    lr_reduced_factor   = 0.1
    lr_reduced_patience = 2

    print('train_Y (trainOrValid) :')
    print(np.shape(train_Y))
    print(np.array(train_Y))
    
    trainModel(model, train_Xs, train_Y, valid_rate, epochs, early_patience, lr_reduced_factor, lr_reduced_patience)
    
    # save model
    if valid == True:
        model.save('valid_' + algo_name)
        
    # predict using model
    if valid == True:
        prediction = model.predict(train_Xs[trainCount:]).flatten()
    else:
        prediction = model.predict(test_Xs).flatten()
            
    prediction = np.clip(prediction, 0.0001, 0.9999)

    print('\n[14] original (normalized and then SIGMOID-ed) prediction:')
    print(np.shape(prediction))
    print(prediction)

    # y -> denormalize(invSigmoid(y))
    for i in range(len(prediction)):
        prediction[i] = invSigmoid(prediction[i])
            
    prediction = prediction * stddev + avg

    # write final submission/prediction
    print('\n[15] converted prediction:')
    print(np.shape(prediction))
    print(prediction)

    prediction = pd.DataFrame(np.reshape(np.array(prediction), (len(prediction), 1)))

    # for valid mode
    if valid == True:
        prediction.to_csv('valid_' + algo_name + '.csv')

    # for test mode
    else:
        prediction.to_csv('test_' + algo_name + '.csv')
        try:
            final_prediction += prediction
        except:
            final_prediction = prediction

    # print final prediction
    if valid == False:
        # final_prediction /= float(count)
        
        print('\n[16] FINAL prediction:')
        print(np.shape(final_prediction))
        print(final_prediction)
        
        return final_prediction
