import pandas as pd
import numpy as np
import math
import csv

if __name__ == '__main__':

    features = 50
    numClass = 4

    # load training data
    trainData = pd.read_csv('train.csv')
    trainFeatures = trainData.loc[:, 'feature_0':'feature_49']
    trainTarget = trainData.loc[:, 'target']
    
    print(trainFeatures)
    print(trainTarget)

    # extract cases where target='Class_1', ..., 'Class_4'
    # for each training feature
    for i in range(features):
        featName = 'feature_' + str(i)
        
        class1 = trainData[trainData['target'] == 'Class_1'].loc[:, featName]
        class2 = trainData[trainData['target'] == 'Class_2'].loc[:, featName]
        class3 = trainData[trainData['target'] == 'Class_3'].loc[:, featName]
        class4 = trainData[trainData['target'] == 'Class_4'].loc[:, featName]
        classes = [class1, class2, class3, class4]

        # distribution

        # average, stddev, min, max, 25%, 50% and 75%
        for j in range(numClass):
            thisClass = classes[j]
            thisClass = pd.Series(thisClass)

            print('\n << feature ' + str(i) + ' class ' + str(j+1) + ' >>')
            print(thisClass.describe())
