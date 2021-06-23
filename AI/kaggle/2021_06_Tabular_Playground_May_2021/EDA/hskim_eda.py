# remove warings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import math
import csv

# get distribution array (the number of items from min to max)
# eg: [0, 1, 3, 1, 2, 7, 2] -> [[0,1], [1,2], [2,2], [3,1], [4,0], [5,0], [6,0], [7,1]]
#     because min=0, max=7, count of 0,1,2,3,4,5,6,7 = 1,2,2,1,0,0,0,1
def getDistribution(arr):

    #minVal = int(min(arr))
    #maxVal = int(max(arr))
    minVal = -8 # use global minimum
    maxVal = 66 # use global maximum

    count_arr = np.zeros((maxVal - minVal + 1, 2))

    for i in range(len(count_arr)):
        count_arr[i][0] = minVal + i

    for i in range(len(arr)):
        count_arr[arr[i] - minVal][1] += 1

    return count_arr

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

    # visualize FEATURE-TARGET INFO ARRAYS
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

        # save and show images
        if info == info0:
            plt.title('average and standard deviation for each class(4) for each feature(50)' + ' '*150, fontsize=10)
            plt.savefig('hskim_eda_avgAndStd.png')
        else: # info1
            plt.title('min / 25% / 50% / 75% / max values for each class(4) for each feature(50)' + ' '*250, fontsize=10)
            plt.savefig('hskim_eda_minMaxAndPercentile.png')
        plt.show()

    # FEATURE DISTRIBUTION arrays
    # print training feature distribution for each target value for each training feature
    for i in range(50):
        featureArr = np.array(trainData.loc[:, 'feature_' + str(i)])
        featureMin = min(featureArr)
        featureMax = max(featureArr)

    # layout of visualization result
    fig, axn = plt.subplots(features, numClass, sharex=True, sharey=True)
    fig.set_size_inches(18, 150)

    # list of FEATURE DISTRIBUTION for each class for each training feature
    distribs = []

    # extract cases where target='Class_1', ..., 'Class_4'
    # for each training feature
    for i in range(features):
        featName = 'feature_' + str(i)
        
        class1 = trainData[trainData['target'] == 'Class_1'].loc[:, featName]
        class2 = trainData[trainData['target'] == 'Class_2'].loc[:, featName]
        class3 = trainData[trainData['target'] == 'Class_3'].loc[:, featName]
        class4 = trainData[trainData['target'] == 'Class_4'].loc[:, featName]
        classes = [class1, class2, class3, class4]

        # find distribution for class 1, 2, 3 and 4
        for j in range(numClass):

            # get distribution array of each class
            thisClass = np.array(classes[j])
            thisClassDistrib = getDistribution(thisClass).astype(int)

            # print
            print('\n << feature ' + str(i) + ' class ' + str(j+1) + ' >>')
            print(thisClassDistrib[:, 0].T)
            print(thisClassDistrib[:, 1].T)

            # append to list of FEATURE DISTRIBUTION
            distribDf = pd.DataFrame(thisClassDistrib)
            distribDf.columns = ['feature value', 'count (feat ' + str(i) + ')']
            
            distribs.append(distribDf)

    # visualize the array
    count = 0
    for ax in axn.flat:
        classNo = count % 4 + 1
        featureNo = count // 4
        print('drawing distribution chart for feature ' + str(featureNo) + ' class ' + str(classNo) + '...')
        
        distrib = distribs[count]
        bp = sns.barplot(x='feature value', y='count (feat ' + str(featureNo) + ')', data=distrib, ax=ax, color='b')

        # axis setting
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
        
        count += 1

    # save and show images 
    plt.title('distribution for each class(4) for each feature(50)' + ' '*200, fontsize=10)
    plt.savefig('hskim_eda_distribution.png')
    plt.show()
