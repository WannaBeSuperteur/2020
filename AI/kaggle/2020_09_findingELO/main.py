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
import compareValid as _CV

# deep Learning
import sys
sys.path.insert(0, '../../../../AI_BASE')
import deepLearning_GPU as DL
import deepLearning_GPU_helper as helper
import deepLearning_main as DLmain
import AIBASE_main as AImain

if __name__ == '__main__':

    #################################
    ###                           ###
    ###    basic configuration    ###
    ###                           ###
    #################################

    # global validation == 0 for all methods
    # global validation > 0 for method 0 and 1

    # define final result (init as None)
    finalResult = None

    # meta info
    trainName = 'TRAIN.txt'
    testName = 'TE_I.txt'
    ftype = 'txt'

    scores = []
    fcolsTrain = ['id', 'input0', 'input1', 'input2', 'input3', 'output']
    fcolsTest = ['id', 'input0', 'input1', 'input2', 'input3']

    # global validation rate is used on method 0, 1, 2, 3 and 4 (that is, except for deep learning)
    globalValidationRate = 0.15 # ** IMPORTANT ** validation rate (if >0, then split training data into training and validation data, randomly)

    validationExceptCols = [5] # except for these columns for validation data
    validationCol = 5 # compare finalResult with column of this index of validation data file (according to targetColName)

    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # validation mode is not available for method 5 and method 6
    method = 0 # ** IMPORTANT ** 0: PCA+kNN, 1: PCA+DT, 2: TextVec+NB, 3: PCA+xgboost, 4: xgboost only, 5: PCA+deep learning, 6: deep learning only

    usePCA = False

    #################################
    ###                           ###
    ###    model configuration    ###
    ###                           ###
    #################################
    
    # make PCA from training data
    PCAdimen = 4 # dimension of PCA
    idCol = 'id'
    targetColName = 'output'
    tfCols = []
    textCols = ['']
    exceptCols = ['id'] # list of columns not used
    exceptColsForMethod2 = ['id'] # list of columns not used for method 2
    exceptTargetForPCA = True # except target column for PCA
    useLog = False # using log for numeric data columns
    logConstant = 10000000 # x -> log2(x + logConstant)
    specificCol = 'dddd' # specific column to solve problem (eg: 'request_text_edit_aware')
    frequentWords = None # frequent words (if not None, do word appearance check)

    # for method 0
    kNN_k = 4 # number k for kNN
    kNN_useAverage = True # use average voting for kNN

    # for method 1 (for only when target value is binary, that is 0 or 1)
    DT_maxDepth = 15 # max depth of decision tree
    DT_criterion = 'entropy' # 'gini' or 'entropy'
    DT_splitter = 'random' # 'best' or 'random'

    # convert to mean of each range when target column is numeric
    DT_numericRange = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

    # for method 3 and 4
    XG_xgBoostLevel = 0 # 0: just execute xgBoost, 1: as https://www.kaggle.com/jatinraina/random-acts-of-pizza-xgboost
    XG_epochs = 100 # xgboost training epochs
    XG_boostRound = 10000 # num_boost_round for xgb.train
    XG_earlyStoppingRounds = 10 # early_stopping_rounds for xgb.train
    XG_foldCount = 5 # XG_foldCount = N -> (N-1)/N of whole dataset is for training, 1/N is for test
    XG_rateOf1s = 0.15 # rate of 1's (using this, portion of 1 should be XG_rateOf1s)
    XG_info = [XG_epochs, XG_boostRound, XG_earlyStoppingRounds, XG_foldCount, XG_rateOf1s]

    # for deep learning (method 5 and 6)
    DL_normalizeTarget = False # normalize training output value? (False because automatically normalized)

    #################################
    ###                           ###
    ### split data into train/val ###
    ###                           ###
    #################################

    trainValid_trainRows = None # rows for training (for train-valid mode where globalValidationRate > 0)
    trainValid_validRows = None # rows for validation (for train-valid mode where globalValidationRate > 0)

    # validation mode & method <= 4 (not deep learning method, returns finalResult)
    if globalValidationRate > 0 and method <= 4:
        testName = trainName # because test (validation) using training data

        # get the number of rows of training data
        ft = open(trainName)
        ftrows = len(ft.readlines())
        ft.close()

        print('ftrows = ' + str(ftrows))

        # specify rows for training and test
        valid = []
        for i in range(ftrows): valid.append(0)

        while sum(valid) < ftrows * globalValidationRate:
            valid[random.randint(0, ftrows-1)] = 1

        # update trainValid_trainRows and trainValid_validRows
        trainValid_trainRows = []
        trainValid_validRows = []
        
        for i in range(ftrows):
            if valid[i] == 1: trainValid_validRows.append(i)
            else: trainValid_trainRows.append(i)

    #################################
    ###                           ###
    ###      model execution      ###
    ###                           ###
    #################################

    # method 0, 1, 3 or 5 -> use PCA    
    if method == 0 or method == 1 or method == 3 or method == 5:

        # set target column name
        targetColumn = 'target'
        if usePCA == False: targetColumn = targetColName

        # get PCA (components and explained variances) for training data
        # df_pca_train: dataFrame with columns including target column [pca0 pca1 ... pcaN target]
        if usePCA == True:

            # obtaining traing data
            print('')
            print('+-----------------------+')
            print('|     Training Data     |')
            print('+-----------------------+')
            (df_pca_train, comp, exva, mean, targetCol) = _PCA.makePCA(trainName, None, trainValid_trainRows, ftype, fcolsTrain, PCAdimen, True, targetColName,
                                                                       tfCols, textCols+exceptCols, None, None, None, exceptTargetForPCA, useLog,
                                                                       logConstant, specificCol, frequentWords)

            # remove target column from comp and mean
            if exceptTargetForPCA == False:
                comp = np.delete(comp, [targetCol], 1)
                mean = np.delete(mean, [targetCol], 0)

            # get PCA (components and explained variances) for test data
            # df_pca_test: dateFrame with columns except for target column [pca0 pca1 ... pcaN]

            # normal mode
            if globalValidationRate == 0 or method == 5:
                print('')
                print('+-----------------------+')
                print('|       Test Data       |')
                print('+-----------------------+')
                (df_pca_test, noUse0, noUse1, noUse2, noUse3) = _PCA.makePCA(testName, None, None, ftype, fcolsTest, PCAdimen, False, None,
                                                                             tfCols, textCols+exceptCols, comp, exva, mean, False, useLog,
                                                                             logConstant, specificCol, frequentWords)

            # validation mode
            else:
                print('')
                print('+-----------------------+')
                print('|    Validation Data    |')
                print('+-----------------------+')
                (df_pca_test, noUse0, noUse1, noUse2, noUse3) = _PCA.makePCA(trainName, validationExceptCols, trainValid_validRows, ftype, fcolsTest, PCAdimen, False, None,
                                                                             tfCols, textCols+exceptCols, comp, exva, mean, False, useLog,
                                                                             logConstant, specificCol, frequentWords)

        # do not use PCA
        else:

            # obtaining traing data
            print('')
            print('+-----------------------+')
            print('|     Training Data     |')
            print('+-----------------------+')
            (df_pca_train, targetCol) = _DF.makeDataFrame(trainName, None, trainValid_trainRows, ftype, fcolsTrain, True, targetColName,
                                                          tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

            # normal mode
            if globalValidationRate == 0 or method == 5:
                print('')
                print('+-----------------------+')
                print('|       Test Data       |')
                print('+-----------------------+')
                (df_pca_test, noUse) = _DF.makeDataFrame(testName, None, None, ftype, fcolsTest, False, targetColName,
                                                     tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

            # validation mode
            else:
                print('')
                print('+-----------------------+')
                print('|    Validation Data    |')
                print('+-----------------------+')
                (df_pca_test, noUse) = _DF.makeDataFrame(trainName, validationExceptCols, trainValid_validRows, ftype, fcolsTest, False, targetColName,
                                                     tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

        # do not use decision tree
        if method == 0:

            # k-NN of test data
            finalResult = _KNN.kNN(df_pca_train, df_pca_test, targetColumn, PCAdimen, kNN_k, kNN_useAverage)

        # use decision tree
        elif method == 1:
            
            # make decision tree
            DT = _DT.createDTfromDF(df_pca_train, PCAdimen, True, DT_maxDepth, DT_criterion, DT_splitter, DT_numericRange)

            # predict test data using decision tree
            finalResult = _DT.predictDT(df_pca_test, DT, True, DT_maxDepth, DT_criterion, DT_splitter)

        # use xgBoost
        # https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
        # if XG_xgBoostLevel=1 then using like https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html
        elif method == 3:
            totalAccuracy = 0 # total accuracy score
            totalROC = 0 # total area under ROC
            times = 20 # number of iterations (tests)

            # iteratively validation
            for i in range(times):
                (accuracy, ROC, _) = _XB.usingXgBoost(df_pca_train, df_pca_test, targetColumn, 'test ' + str(i), True, True, XG_xgBoostLevel, XG_info)
                totalAccuracy += accuracy
                totalROC += ROC

            print('\n<<< [22] total accuracy and ROC for xgBoost >>>')
            print('avg accuracy : ' + str(totalAccuracy / times))
            print('avg ROC      : ' + str(totalROC / times))

            # predict values
            finalResult = _XB.usingXgBoost(df_pca_train, df_pca_test, targetColumn, 'test ' + str(i), True, False, XG_xgBoostLevel, XG_info)

        # use Deep Learning for PCA-ed data
        elif method == 5:
            
            # save data as file
            DLmain.dataFromDF(df_pca_train, df_pca_test, targetColumn, [], DL_normalizeTarget)

            # deep learning
            AImain.AIbase_deeplearning()
            
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
        print('')
        print('+-----------------------+')
        print('|     Training Data     |')
        print('+-----------------------+')
        (train_df, targetColOfTrainDataFrame) = _DF.makeDataFrame(trainName, None, trainValid_trainRows, ftype, fcolsTrain, True, targetColName,
                                                                  tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)

        if globalValidationRate == 0: # normal mode
            print('')
            print('+-----------------------+')
            print('|       Test Data       |')
            print('+-----------------------+')
            (test_df, noUse) = _DF.makeDataFrame(testName, None, None, ftype, fcolsTest, False, targetColName,
                                                 tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)
        else: # validation mode
            print('')
            print('+-----------------------+')
            print('|    Validation Data    |')
            print('+-----------------------+')
            (test_df, noUse) = _DF.makeDataFrame(trainName, validationExceptCols, trainValid_validRows, ftype, fcolsTest, False, targetColName,
                                                 tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)

        # print dataFrames
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
        # if method == 2 and target value is not binary(0 or 1), it may cause error
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

    # method 4 or method 6 (xgboost only or deep learning)
    if method == 4 or method == 6:

        # get train and test dataFrame
        print('')
        print('+-----------------------+')
        print('|     Training Data     |')
        print('+-----------------------+')
        (df_pca_train, targetColOfTrainDataFrame) = _DF.makeDataFrame(trainName, None, trainValid_trainRows, ftype, fcolsTrain, True, targetColName,
                                                                      tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

        if globalValidationRate == 0 or method == 6: # normal mode
            print('')
            print('+-----------------------+')
            print('|       Test Data       |')
            print('+-----------------------+')
            (df_pca_test, noUse) = _DF.makeDataFrame(testName, None, None, ftype, fcolsTest, False, targetColName,
                                                     tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)
        else: # validation mode
            print('')
            print('+-----------------------+')
            print('|    Validation Data    |')
            print('+-----------------------+')
            (df_pca_test, noUse) = _DF.makeDataFrame(trainName, validationExceptCols, trainValid_validRows, ftype, fcolsTest, False, targetColName,
                                                     tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

    # xgboost only
    # if XG_xgBoostLevel = 1 then as https://www.kaggle.com/jatinraina/random-acts-of-pizza-xgboost
    #                          (ref: https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html)
    if method == 4:

        # print training and test data
        print('\n<<< [32] df_pca_train method==4 >>>')
        print(df_pca_train)

        print('\n<<< [33] df_pca_test method==4 >>>')
        print(df_pca_test)

        # run xgboost
        # when setting validation as True, finally, always return error at [33] (both XG_xgBoostLevel=0 and XG_xgBoostLevel=1)
        finalResult = _XB.usingXgBoost(df_pca_train, df_pca_test, targetColName, 'method4', True, False, XG_xgBoostLevel, XG_info)

        print('\n<<< [34] len of finalResult >>>')
        print(len(finalResult))

    # use Deep Learning
    elif method == 6:

        # print training and test data
        print('\n<<< [35] df_pca_train >>>')
        print(df_pca_train)

        print('\n<<< [36] df_pca_test >>>')
        print(df_pca_test)

        # save data as file
        DLmain.dataFromDF(df_pca_train, df_pca_test, targetColName, exceptCols, DL_normalizeTarget)

        # deep learning
        AImain.AIbase_deeplearning()

    #################################
    ###                           ###
    ###        model test         ###
    ###                           ###
    #################################

    # exit if final result does not exist (method 5 or method 6)
    if method == 5 or method == 6:
        print('\nfinish')
        exit()

    # for method 5 and method 6, this causes error because finalResult is None
    print('\n<<< [37] final result (len=' + str(len(finalResult)) + ') >>>')
    print(np.array(finalResult))

    # write result if final result exists
    result = str(targetColName) + '\n'
    for i in range(len(finalResult)): result += str(finalResult[i]) + '\n'

    # write the result to file
    f = open('result.csv', 'w')
    f.write(result)
    f.close()

    # compare final result with validation data
    if globalValidationRate > 0:
        _CV.compare(finalResult, testName, validationCol, trainValid_validRows)

# 향후계획:
# 딥러닝을 제외한 모든 머신러닝 알고리즘에 대해 Training data (train_df) 를 train_df와 valid_df로 구분하여 성능 평가 (ING)
#     train_df와 valid_df로 구분, normal mode method=0,1,5,6 및 validation mode method=0,1 에서 작동 확인 (FIN)
#     성능 평가 라이브러리를 작성하여 result.csv와 실제 데이터를 비교, 성능 출력 (FIN)
#     Decision Tree 알고리즘에서 학습 전에 데이터를 카테고리화 (예: 100 단위로 반올림) 적용 (FIN)
#     보다 간단하고 규칙성 있는 dataset을 이용하여 재실험 (normal 0,1,5,6 and valid 0,1) 및 문제 해결 (ING)
#     PCA 없이 method 0,1,3,5 실행 (normal 0,1,5 and valid 0,1) (ING)
#     method 2,4,6에서 validation dataset을 trainName에서 가져온다고 설정했으므로 재실험 (valid 6) (fcolsTest에서 id col 제외) (ING)
