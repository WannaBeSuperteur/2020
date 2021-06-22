# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import csv

if __name__ == '__main__':

    # dataset metadata
    features = 50
    numClass = 4

    # load training data
    trainData = pd.read_csv('train.csv')
    trainFeatures = trainData.loc[:, 'feature_0':'feature_49']
    trainTarget = trainData.loc[:, 'target']
    
    print(trainFeatures)
    print(trainTarget)

    # FEATURE-TARGET INFO ARRAYS
    # average, stddev, min, max, 25%, 50% and 75%
    # for each target value for each training feature
    averages = np.zeros((features, numClass))
    stddevs = np.zeros((features, numClass))
    mins = np.zeros((features, numClass))
    q1s = np.zeros((features, numClass))
    q2s = np.zeros((features, numClass))
    q3s = np.zeros((features, numClass))
    maxs = np.zeros((features, numClass))

    # extract cases where target='Class_1', ..., 'Class_4'
    # for each training feature
    for i in range(features):
        featName = 'feature_' + str(i)
        
        class1 = trainData[trainData['target'] == 'Class_1'].loc[:, featName]
        class2 = trainData[trainData['target'] == 'Class_2'].loc[:, featName]
        class3 = trainData[trainData['target'] == 'Class_3'].loc[:, featName]
        class4 = trainData[trainData['target'] == 'Class_4'].loc[:, featName]
        classes = [class1, class2, class3, class4]

        # find and print average, stddev, min, max, 25%, 50% and 75%
        for j in range(numClass):
            thisClass = classes[j]
            thisClass = pd.Series(thisClass)

            # average and stddev
            avg = round(thisClass.mean(), 4)
            std = round(thisClass.std(), 4)

            # min, 25%, 50%, 75% and max
            min_ = round(float(thisClass.min()), 4)
            q1 = round(thisClass.quantile(0.25), 4)
            q2 = round(thisClass.quantile(0.5), 4)
            q3 = round(thisClass.quantile(0.75), 4)
            max_ = round(float(thisClass.max()), 4)

            # print
            print('\n << feature ' + str(i) + ' class ' + str(j+1) + ' >>')
            print(avg, std, min_, q1, q2, q3, max_)

            # insert into FEATURE-TARGET INFO ARRAYS
            averages[i][j] = avg
            stddevs[i][j] = std
            mins[i][j] = min_
            q1s[i][j] = q1 # quarter 1 (25%)
            q2s[i][j] = q2 # quarter 2 (50%, median)
            q3s[i][j] = q3 # quarter 3 (75%)
            maxs[i][j] = max_

    # print FEATURE-TARGET INFO ARRAYS
    info0 = [averages, stddevs]
    info1 = [mins, q1s, q2s, q3s, maxs]

    for info in [info0, info1]:
        
        fig, axn = plt.subplots(1, len(info), sharex=True, sharey=True)
        fig.set_size_inches(18, 10)
        
        count = 0
        for ax in axn.flat:
            hm = sns.heatmap(info[count], ax=ax, cmap='RdYlGn_r', annot=True, linewidths=0.05,
                             annot_kws={"size":8})
            count += 1

        # show images
        if info == info0:
            plt.title('average and standard deviation for each class(4) for each feature(50)' + ' '*150, fontsize=10)
            plt.savefig('hskim_eda_avgAndStd.png')
        else: # info1
            plt.title('min / 25% / 50% / 75% / max values for each class(4) for each feature(50)' + ' '*250, fontsize=10)
            plt.savefig('hskim_eda_minMaxAndPercentile.png')
        plt.show()

    # FEATURE DISTRIBUTION arrays
    # print training feature distribution for each target value for each training feature

    # add distribution into the picture
