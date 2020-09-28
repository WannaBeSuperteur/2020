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
from nltk.stem import WordNetLemmatizer # wml
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

if __name__ == '__main__':

    # meta info
    trainName = 'train.json'
    testName = 'test.json'

    #################################
    ###                           ###
    ###    model configuration    ###
    ###                           ###
    #################################
    
    # make PCA from training data
    PCAdimen = 4 # dimension of PCA
    idCol = 'request_id'
    targetColName = 'requester_received_pizza'
    tfCols = ['post_was_edited', 'requester_received_pizza']
    textCols = ['request_text', 'request_text_edit_aware',
                    'request_title', 'requester_subreddits_at_request', 'requester_user_flair']
    exceptCols = ["giver_username_if_known",
                  "request_id",
                  "number_of_downvotes_of_request_at_retrieval",
                  "number_of_upvotes_of_request_at_retrieval",
                  "post_was_edited",
                  "request_number_of_comments_at_retrieval",
                  "request_text",
                  "requester_account_age_in_days_at_retrieval",
                  "requester_days_since_first_post_on_raop_at_retrieval",
                  "requester_number_of_comments_at_retrieval",
                  "requester_number_of_comments_in_raop_at_retrieval",
                  "requester_number_of_posts_at_retrieval",
                  "requester_number_of_posts_on_raop_at_retrieval",
                  "requester_subreddits_at_request",
                  "requester_upvotes_minus_downvotes_at_retrieval",
                  "requester_upvotes_plus_downvotes_at_retrieval",
                  "requester_user_flair",
                  "requester_username",
                  "unix_timestamp_of_request_utc"] # list of columns not used
    exceptColsForMethod2 = ["giver_username_if_known", "request_id", "requester_username"] # list of columns not used for method 2
    exceptTargetForPCA = True # except target column for PCA
    useLog = False # using log for numeric data columns
    logConstant = 10000000 # x -> log2(x + logConstant)
    specificCol = None # specific column to solve problem (eg: 'request_text_edit_aware')
    frequentWords = None # frequent words (if not None, do word appearance check)

    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    method = 0 # 0: PCA+kNN, 1: PCA+DT, 2: TextVec+NB, 3: PCA+xgboost, 4: xgboost only

    # for method 0
    kNN_k = 120 # number k for kNN
    useAverage = True # use average voting for kNN

    # for method 2
    DT_maxDepth = 15 # max depth of decision tree
    DT_criterion = 'entropy' # 'gini' or 'entropy'
    DT_splitter = 'random' # 'best' or 'random'

    # for method 4
    xgBoostLevel = 1 # 0: just execute xgBoost, 1: as https://www.kaggle.com/jatinraina/random-acts-of-pizza-xgboost
    epochs = 100
    boostRound = 10000
    earlyStoppingRounds = 10
    foldCount = 5
    rateOf1s = 0.15 # rate of 1's (using this, portion of 1 should be rateOf1s)
    info = [epochs, boostRound, earlyStoppingRounds, foldCount, rateOf1s]

    #################################
    ###                           ###
    ###      model execution      ###
    ###                           ###
    #################################

    # method 0, 1 or 3 -> use PCA    
    if method == 0 or method == 1 or method == 3:

        # get PCA (components and explained variances) for training data
        # df_pca_train: dataFrame with columns including target column [pca0 pca1 ... pcaN target]
        (df_pca_train, comp, exva, mean, targetCol) = makePCA(trainName, PCAdimen, True, targetColName, tfCols, textCols+exceptCols,
                                                              None, None, None, exceptTargetForPCA, useLog, logConstant, specificCol, frequentWords)

        # remove target column from comp and mean
        if exceptTargetForPCA == False:
            comp = np.delete(comp, [targetCol], 1)
            mean = np.delete(mean, [targetCol], 0)

        # get PCA (components and explained variances) for test data
        # df_pca_test: dateFrame with columns except for target column [pca0 pca1 ... pcaN]
        (df_pca_test, noUse0, noUse1, noUse2, noUse3) = makePCA(testName, PCAdimen, False, None, tfCols, textCols+exceptCols,
                                                                comp, exva, mean, False, useLog, logConstant, specificCol, frequentWords)

        # do not use decision tree
        if method == 0:

            # k-NN of test data
            finalResult = kNN(df_pca_train, df_pca_test, 'target', PCAdimen, kNN_k, useAverage)

        # use decision tree
        elif method == 1:
            
            # make decision tree
            DT = createDTfromDF(df_pca_train, PCAdimen, True, DT_maxDepth, DT_criterion, DT_splitter)

            # predict test data using decision tree
            finalResult = predictDT(df_pca_test, DT, True, DT_maxDepth, DT_criterion, DT_splitter)

        # use xgBoost
        # https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
        # if xgBoostLevel=1 then using like https://swlock.blogspot.com/2019/02/xgboost-stratifiedkfold-kfold.html
        elif method == 3:
            totalAccuracy = 0 # total accuracy score
            totalROC = 0 # total area under ROC
            times = 20 # number of iterations (tests)

            # iteratively validation
            for i in range(times):
                (accuracy, ROC, _) = usingXgBoost(df_pca_train, df_pca_test, 'target', 'test ' + str(i), True, True, xgBoostLevel, info)
                totalAccuracy += accuracy
                totalROC += ROC

            print('\n<<< [22] total accuracy and ROC for xgBoost >>>')
            print('avg accuracy : ' + str(totalAccuracy / times))
            print('avg ROC      : ' + str(totalROC / times))

            # predict values
            finalResult = usingXgBoost(df_pca_train, df_pca_test, 'target', 'test ' + str(i), True, False, xgBoostLevel, info)

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
        count_vect = CountVectorizer(analyzer=preprocess_text)

        # get train and test dataFrame
        (train_df, targetColOfTrainDataFrame) = makeDataFrame(trainName, True, targetColName, tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)
        (test_df, noUse) = makeDataFrame(testName, False, targetColName, tfCols, exceptColsForMethod2, useLog, logConstant, specificCol, frequentWords)

        print('\n<<< [23] train_df.columns >>>')
        print(train_df.columns)
        print('\n<<< [24] train_df >>>')
        print(train_df)
        print('\n<<< [25] test_df.columns >>>')
        print(test_df.columns)
        print('\n<<< [26] test_df >>>')
        print(test_df)

        print('\n<<< [27] train_df[' + targetColName + '] >>>')
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

        print('\n<<< [28] trainSet >>>')
        print(trainSet)
        print('\n<<< [29] trainTags >>>')
        print(trainTags)

        # In [53] / In [55]:
        clf = MultinomialNB()
        clf.fit(trainSet, trainTags)

        # In [56]:
        predictions = clf.predict(testSet)

        print('\n<<< [30] predictions >>>')
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
        (df_pca_train, targetColOfTrainDataFrame) = makeDataFrame(trainName, True, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)
        (df_pca_test, noUse) = makeDataFrame(testName, False, targetColName, tfCols, textCols+exceptCols, useLog, logConstant, specificCol, frequentWords)

        # print training and test data
        print('\n<<< [31] df_pca_train method==4 >>>')
        print(df_pca_train)

        print('\n<<< [32] df_pca_test method==4 >>>')
        print(df_pca_test)

        # run xgboost
        # when setting validation as True, finally, always return error at [33] (both xgBoostLevel=0 and xgBoostLevel=1)
        finalResult = usingXgBoost(df_pca_train, df_pca_test, targetColName, 'method4', True, False, xgBoostLevel, info)

        print('\n<<< [33] len of finalResult >>>')
        print(len(finalResult))

    # write result
    jf = open(testName, 'r')
    json_file = jf.read()
    jf.close()

    json_data = json.loads(json_file)
    
    result = str(idCol) + ',' + str(targetColName) + '\n'
    for i in range(len(json_data)):
        x = json_data[i][idCol]
        result += str(x) + ',' + str(finalResult[i]) + '\n'

    f = open('result.csv', 'w')
    f.write(result)
    f.close()
