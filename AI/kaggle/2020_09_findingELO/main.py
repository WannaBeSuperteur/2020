import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab
import json
import math
import re
import random

# to make decision tree
# https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

# for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import export_text

# for vectorization and naive bayes model
# source: https://www.kaggle.com/alvations/basic-nlp-with-nltk/
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import webtext
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer # porter
from nltk.stem import WordNetLemmatizer # wnl
from nltk import pos_tag
from collections import Counter
from io import StringIO
from operator import itemgetter
from sklearn.model_selection import train_test_split, StratifiedKFold

# using xgboost
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score

import decisionTree as _DT
import frequentWord as _FW
import kNN as _KNN
import makeDF as _DF
import makePCA as _PCA
import printData as _PD
import xgBoost as _XB
import textVec as _TV

# deep Learning
import deepLearning_main as DL
import deepLearning_GPU_helper as helper

if __name__ == '__main__':

    # meta info
    trainName = 'data_trainPgn.txt'
    testName = 'data_testPgn.txt'
    ftype = 'txt'
    fcolsTrain = ['id', 'result', 'welo', 'belo',
                  'score0', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9',
                  'score10', 'score11', 'score12', 'score13', 'score14', 'score15', 'score16', 'score17', 'score18', 'score19',
                  'score20', 'score21', 'score22', 'score23', 'score24', 'score25', 'score26', 'score27', 'score28', 'score29']
    fcolsTest = ['id', 'result',
                 'score0', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9',
                 'score10', 'score11', 'score12', 'score13', 'score14', 'score15', 'score16', 'score17', 'score18', 'score19',
                 'score20', 'score21', 'score22', 'score23', 'score24', 'score25', 'score26', 'score27', 'score28', 'score29']

    #################################
    ###                           ###
    ###    model configuration    ###
    ###                           ###
    #################################
    
    # make PCA from training data
    PCAdimen = 4 # dimension of PCA
    idCol = 'id'
    targetColName = 'belo'
    tfCols = []
    textCols = ['result']
    exceptCols = ['id', 'welo'] # list of columns not used
    exceptColsForMethod2 = ['id', 'welo'] # list of columns not used for method 2
    exceptTargetForPCA = True # except target column for PCA
    useLog = False # using log for numeric data columns
    logConstant = 10000000 # x -> log2(x + logConstant)
    specificCol = 'score0' # specific column to solve problem (eg: 'request_text_edit_aware')
    frequentWords = None # frequent words (if not None, do word appearance check)

    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    method = 6 # 0: PCA+kNN, 1: PCA+DT, 2: TextVec+NB, 3: PCA+xgboost, 4: xgboost only

    # for method 0
    kNN_k = 120 # number k for kNN
    useAverage = True # use average voting for kNN

    # for method 2
    DT_maxDepth = 15 # max depth of decision tree
    DT_criterion = 'entropy' # 'gini' or 'entropy'
    DT_splitter = 'random' # 'best' or 'random'

    # for method 4
    xgBoostLevel = 0 # 0: just execute xgBoost, 1: as https://www.kaggle.com/jatinraina/random-acts-of-pizza-xgboost
    epochs = 100
    boostRound = 10000
    earlyStoppingRounds = 10
    foldCount = 5
    rateOf1s = 0.15 # rate of 1's (using this, portion of 1 should be rateOf1s)
    info = [epochs, boostRound, earlyStoppingRounds, foldCount, rateOf1s]

    # for deep learning (method 5 and 6)
    deepLearningFn = ['data_train_i.txt', 'data_train_o.txt', 'data_test_i.txt', 'data_test_o.txt']
    valid = 0.25 # portion of valid data (0.0 for training, >0.0 for training->validation)
    normalizeTarget = True # normalize training output value?

    #################################
    ###                           ###
    ###      model execution      ###
    ###                           ###
    #################################

    # method 0, 1, 3 or 5 -> use PCA    
    if method == 0 or method == 1 or method == 3 or method == 5:

        # get PCA (components and explained variances) for training data
        # df_pca_train: dataFrame with columns including target column [pca0 pca1 ... pcaN target]
        (df_pca_train, comp, exva, mean, targetCol) = _PCA.makePCA(trainName, ftype, fcolsTrain, PCAdimen, True, targetColName, tfCols, textCols+exceptCols,
                                                              None, None, None, exceptTargetForPCA, useLog, logConstant, specificCol, frequentWords)

        # remove target column from comp and mean
        if exceptTargetForPCA == False:
            comp = np.delete(comp, [targetCol], 1)
            mean = np.delete(mean, [targetCol], 0)

        # get PCA (components and explained variances) for test data
        # df_pca_test: dateFrame with columns except for target column [pca0 pca1 ... pcaN]
        (df_pca_test, noUse0, noUse1, noUse2, noUse3) = _PCA.makePCA(testName, ftype, fcolsTest, PCAdimen, False, None, tfCols, textCols+exceptCols,
                                                                comp, exva, mean, False, useLog, logConstant, specificCol, frequentWords)

        # do not use decision tree
        if method == 0:

            # k-NN of test data
            finalResult = _KNN.kNN(df_pca_train, df_pca_test, 'target', PCAdimen, kNN_k, useAverage)

        # use decision tree
        elif method == 1:
            
            # make decision tree
            DT = _DT.createDTfromDF(df_pca_train, PCAdimen, True, DT_maxDepth, DT_criterion, DT_splitter)

            # predict test data using decision tree
            finalResult = _DT.predictDT(df_pca_test, DT, True, DT_maxDepth, DT_criterion, DT_splitter)

        # use xgBoost
        # https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
        # if xgBoostLevel=1 then using like https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html
        elif method == 3:
            totalAccuracy = 0 # total accuracy score
            totalROC = 0 # total area under ROC
            times = 20 # number of iterations (tests)

            # iteratively validation
            for i in range(times):
                (accuracy, ROC, _) = _XB.usingXgBoost(df_pca_train, df_pca_test, 'target', 'test ' + str(i), True, True, xgBoostLevel, info)
                totalAccuracy += accuracy
                totalROC += ROC

            print('\n<<< [22] total accuracy and ROC for xgBoost >>>')
            print('avg accuracy : ' + str(totalAccuracy / times))
            print('avg ROC      : ' + str(totalROC / times))

            # predict values
            finalResult = _XB.usingXgBoost(df_pca_train, df_pca_test, 'target', 'test ' + str(i), True, False, xgBoostLevel, info)

        # use Deep Learning for PCA-ed data
        elif method == 5:
            
            # save data as file
            DL.dataFromDF(df_pca_train, df_pca_test, 'target', [], deepLearningFn, normalizeTarget)

            # deep learning procedure (using file and valid value)
            finalResult = DL.deepLearningProcedure(deepLearningFn, valid, normalizeTarget)

            # print final result
            # "if valid > 0, ERROR is occurred"
            print('\n<<< [23] len of finalResult >>>')
            print(len(finalResult))
            
    # method 2 -> do not use Decision Tree, use text vectorization + Naive Bayes
    # source: https://www.kaggle.com/alvations/basic-nlp-with-nltk
    elif method == 2:

        if specificCol == None:
            print('specificCol must be specified')
            exit()
        
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')

        # create count vectorizer
        count_vect = CountVectorizer(analyzer=_TV.preprocess_text)

        # get train and test dataFrame
        (train_df, targetColOfTrainDataFrame) = _DF.makeDataFrame(trainName, ftype, fcolsTrain, True, targetColName, tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)
        (test_df, noUse) = _DF.makeDataFrame(testName, ftype, fcolsTest, False, targetColName, tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)

        print('\n<<< [24] train_df.columns >>>')
        print(train_df.columns)
        print('\n<<< [25] train_df >>>')
        print(train_df)
        print('\n<<< [26] test_df.columns >>>')
        print(test_df.columns)
        print('\n<<< [27] test_df >>>')
        print(test_df)

        print('\n<<< [28] train_df[' + targetColName + '] >>>')
        print(train_df[targetColName])

        # change train_df['requester_received_pizza']
        for i in range(len(train_df)):
            if train_df.at[i, targetColName] == 1:
                train_df.at[i, targetColName] = True
            else:
                train_df.at[i, targetColName] = False

        # fit transform each column for training data
        # In [51] and In [52] / In [55]:
        trainSet = count_vect.fit_transform(train_df[specificCol])
        trainTags = train_df[targetColName].astype('bool')
        testSet = count_vect.transform(test_df[specificCol])

        print('\n<<< [29] trainSet >>>')
        print(trainSet)
        print('\n<<< [30] trainTags >>>')
        print(trainTags)

        # In [53] / In [55]:
        clf = MultinomialNB()
        clf.fit(trainSet, trainTags)

        # In [56]:
        predictions = clf.predict(testSet)

        print('\n<<< [31] predictions >>>')
        print(predictions)

        # create finalResult based on predictions
        finalResult = []
        for i in range(len(predictions)):
            if predictions[i] == True: finalResult.append(1)
            else: finalResult.append(0)

    # xgboost only
    # if xgBoostLevel = 1 then as https://www.kaggle.com/jatinraina/random-acts-of-pizza-xgboost
    #                          (ref: https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html)
    elif method == 4:
        
        # get train and test dataFrame
        (df_pca_train, targetColOfTrainDataFrame) = _DF.makeDataFrame(trainName, ftype, fcolsTrain, True, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)
        (df_pca_test, noUse) = _DF.makeDataFrame(testName, ftype, fcolsTest, False, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

        # print training and test data
        print('\n<<< [32] df_pca_train method==4 >>>')
        print(df_pca_train)

        print('\n<<< [33] df_pca_test method==4 >>>')
        print(df_pca_test)

        # run xgboost
        # when setting validation as True, finally, always return error at [33] (both xgBoostLevel=0 and xgBoostLevel=1)
        finalResult = _XB.usingXgBoost(df_pca_train, df_pca_test, targetColName, 'method4', True, False, xgBoostLevel, info)

        print('\n<<< [34] len of finalResult >>>')
        print(len(finalResult))

    # use Deep Learning
    elif method == 6:

        # get train and test dataFrame
        (df_pca_train, targetColOfTrainDataFrame) = _DF.makeDataFrame(trainName, ftype, fcolsTrain, True, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)
        (df_pca_test, noUse) = _DF.makeDataFrame(testName, ftype, fcolsTest, False, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

        # print training and test data
        print('\n<<< [35] df_pca_train method==4 >>>')
        print(df_pca_train)

        print('\n<<< [36] df_pca_test method==4 >>>')
        print(df_pca_test)

        # save data as file
        DL.dataFromDF(df_pca_train, df_pca_test, targetColName, exceptCols, deepLearningFn, normalizeTarget)

        # deep learning procedure (using file and valid value)
        finalResult = DL.deepLearningProcedure(deepLearningFn, valid, normalizeTarget)

        # print final result
        # "if valid > 0, ERROR is occurred"
        print('\n<<< [37] len of finalResult >>>')
        print(len(finalResult))

    # write result
    result = str(idCol) + ',' + str(targetColName) + '\n'
    
    if ftype == 'json': # type is 'json'
        jf = open(testName, 'r')
        json_file = jf.read()
        jf.close()

        json_data = json.loads(json_file)

        # make result for each line
        for i in range(len(json_data)):
            x = json_data[i][idCol]
            result += str(x) + ',' + str(finalResult[i]) + '\n'

    elif ftype == 'csv': # type is 'csv'
        csv_df = pd.read_csv(fn)

        # make result for each line
        for i in range(len(csv_df)):
            x = csv_df[i][idCol]
            result += str(x) + ',' + str(finalResult[i]) + '\n'

    elif ftype == 'txt': # type is 'txt'
        f_ = open(testName, 'r')
        fl = f_.readlines()
        f_.close()

        # find the index of idcol
        idColIndex = 0
        for i in range(len(fcolsTest)):
            if fcolsTest[i] == idCol:
                idColIndex = i
                break

        # make result for each line
        for i in range(len(fl)):
            x = fl[i].split('\t')[0]
            result += str(x) + ',' + str(finalResult[i]) + '\n'

    # write the result to file
    f = open('result.csv', 'w')
    f.write(result)
    f.close()
