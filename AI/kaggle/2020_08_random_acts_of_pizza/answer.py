import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab
import json
import math

# to make decision tree
# https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

# for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# print data point as 2d or 3d space
# data should be PCA-ed data
def printDataAsSpace(n_cols, df_pca):

    # set markers
    markers = ['s', 'o']

    # plot if the value of n_cols is 2
    # https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
    if n_cols == 2:

        # add each point
        for i, marker in enumerate(markers):
            x_axis_data = df_pca[df_pca['target']==i]['pca0']
            y_axis_data = df_pca[df_pca['target']==i]['pca1']
            plt.scatter(x_axis_data, y_axis_data, marker=marker, label=df_pca['target'][i])

        # set labels and show
        plt.legend()
        plt.xlabel('pca0')
        plt.ylabel('pca1')
        plt.show()

    # plot in the space, if the value of n_cols is 3
    # https://python-graph-gallery.com/372-3d-pca-result/
    elif n_cols == 3:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca['pca0'], df_pca['pca1'], df_pca['pca2'], c=df_pca['target'])

        # set labels and show
        ax.set_xlabel('pca0')
        ax.set_ylabel('pca1')
        ax.set_zlabel('pca2')
        plt.show()

    else: print('n_cols should be 2 or 3 to print data as 2d/3d space')

# fn        : file name
# n_cols    : number of columns(components) of PCA
# isTrain   : training(True) or not(False)
# target    : target column if isTrain is True
# tfCols    : columns that contains True or False values
# textCols  : columns that contains text values
# exceptCols: using the columns except for them
# comp      : components of PCA used
# exva      : explained variances of PCA used
# mean      : mean of PCA used
def makePCA(fn, n_cols, isTrain, target, tfCols, textCols, exceptCols, comp, exva, mean):
    print('\n ######## makePCA function ########')

    # open and show plt data
    jf = open(fn, 'r')
    json_loaded = json.load(jf)
    json_data = pd.DataFrame(json_loaded)
    targetCol = -1 # index of target column

    # True to 1, False to 0, and decimal to 0
    # https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false
    tfColsIndex = []
    for i in range(len(tfCols)): tfColsIndex.append(-1) # column indices of tfCols

    # initialize tfColsIndex
    for i in range(len(json_data.columns)):
        for j in range(len(tfCols)):
            if json_data.columns[i] == tfCols[j]: tfColsIndex[j] = i

    # extract column name before change into np.array
    dataCols = np.array(json_data.columns)

    # change json_data into np.array
    json_data = json_data.to_numpy()

    # modify values: True to 1, False to 0, and decimal to 0
    for x in range(2):

        # True to 1, False to 0, and decimal to 0
        for i in range(len(json_data)):
            if str(json_data[i][tfColsIndex[x]]) == 'True':
                json_data[i][tfColsIndex[x]] = 1
            else:
                json_data[i][tfColsIndex[x]] = 0

    json_data = pd.DataFrame(json_data, columns=dataCols)
    print('<<< [0] json_data.shape >>>\n' + str(json_data.shape))

    # create data
    # .data and .target
    if isTrain == True: targetCol = target # target column name
    
    dataPart = [] # data columns
    if isTrain == True: targetPart = [] # target column
    extractCols = [] # extracted columns = dataPart + targetPart
    extractColInfo = [] # info about extracted columns (type, etc.)

    for col in dataCols:

        # except for these columns
        continueThis = False
        for i in range(len(exceptCols)):
            if col == exceptCols[i]:
                continueThis = True
                break
        if continueThis == True: continue
        
        dataPartAdded = False

        # accept columns only if not all values are the same (then not meaningful)
        # so check if max(col) > min(col)
        
        # not in targetCol and all of values are numeric -> dataPart
        if isTrain == True: # train mode -> targetCol exists
            if col != targetCol and not col in textCols:
                if max(json_data[col]) > min(json_data[col]):
                    dataPart.append(col)
                    extractCols.append(col)
                    extractColInfo.append('data')
                    dataPartAdded = True

        else: # test mode -> targetCol does not exist
            if not col in textCols:
                if max(json_data[col]) > min(json_data[col]):
                    dataPart.append(col)
                    extractCols.append(col)
                    extractColInfo.append('data')
                    dataPartAdded = True

        # if equal to targetCol
        if isTrain == True and dataPartAdded == False:
            if col == targetCol:
                targetPart.append(col)
                extractCols.append(col)
                extractColInfo.append('target')

                # set index to the index of target column
                targetCol = len(extractCols)-1

    print('\n<<< [1] dataPart and extractCols >>>')
    for i in range(len(dataPart)): print(dataPart[i])
    print('')
    for i in range(len(extractCols)): print(extractCols[i] + ' : ' + extractColInfo[i])

    # bind the data and target
    if isTrain == True: dataSet = {'data':json_data[dataPart], 'target':json_data[targetPart]}
    else: dataSet = {'data':json_data[dataPart]}
    
    dataSetDF = json_data[extractCols]
    dataSetDF = dataSetDF.astype(float) # change dataSetDF into float type

    # display correlation
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    print('\n<<< [2] dataSetDF >>>')
    print(dataSetDF)

    df = dataSetDF.corr() # get correlation
    seab.clustermap(df,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # to standard normal distribution
    scaled = StandardScaler().fit_transform(dataSetDF)
    
    # PCA
    # https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
    initializePCA = False # initializing PCA?
    
    if str(comp) == 'None' or str(exva) == 'None' or str(mean) == 'None': # create PCA if does not exist
        pca = PCA(n_components=n_cols)
        pca.fit(scaled)

        # get components and explained variances of PCA
        comp = pca.components_
        exva = pca.explained_variance_
        mean = pca.mean_
        
        initializePCA = True

    # https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    # print pca.components_ and pca.explained_variance_
    print('\n<<< [3] pca.components_ >>>\n' + str(comp))
    print('\n<<< [4] pca.explained_variance_ >>>\n' + str(exva))
    print('\n<<< [5] pca.mean_ >>>\n' + str(mean))

    # create PCA using comp and exva
    if initializePCA == False:
        pca = PCA(n_components=n_cols)
        pca.components_ = comp
        pca.explained_variance_ = exva
        pca.mean_ = mean
    
    # apply PCA to the data
    scaledPCA = pca.transform(scaled)

    print('\n<<< [6] scaledPCA.shape >>>\n' + str(scaledPCA.shape))
    print('\n<<< [7] scaledPCA.data.shape >>>\n' + str(scaledPCA.data.shape))

    print('\n<<< [8] scaledPCA >>>')
    print(scaledPCA)

    # name each column for PCA transformed data
    pca_cols = []
    for i in range(n_cols): pca_cols.append('pca' + str(i))
    df_pca = pd.DataFrame(scaledPCA, columns=pca_cols)
    if isTrain == True: df_pca['target'] = dataSetDF[target]

    print('\n<<< [9] df_pca >>>')
    print(df_pca)

    df_pcaCorr = df_pca.corr()
    seab.clustermap(df_pcaCorr,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # immediately return the pca if testing
    if isTrain == False: return (df_pca, comp, exva, mean, targetCol)

    # print data as 2d or 3d space
    printDataAsSpace(n_cols, df_pca)

    return (df_pca, comp, exva, mean, targetCol)

# k Nearest Neighbor algorithm
# dfTrain    : dataframe for training data
# dfTest     : dataframe for test data
# targetCol  : target column for dfTrain
# targetIndex: index for the target column
# k          : number of neighbors
def kNN(dfTrain, dfTest, targetCol, targetIndex, k):
    print('\n ######## kNN function ########')

    # count of each value for training data
    targetVals = list(set(dfTrain[targetCol].values)) # set of target values
    classCount = dfTrain[targetCol].value_counts() # class count for each target value
    print(classCount)
    print(classCount[0.0])

    # convert to numpy array
    dfTrain = np.array(dfTrain)
    dfTest = np.array(dfTest)
    print('<<< [0] dfTrain >>>')
    print(dfTrain)
    print('\n<<< [1] dfTest >>>')
    print(dfTest)

    # kNN classification result
    result = []
    resultT = [] # transport of result

    # mark the result using k-nearest neighbor
    for i in range(len(dfTest)): # for each test data
        if i % 10 == 0: print('test data ' + str(i))
        
        thisTestData = dfTest[i]

        # [distance between the test data and each training data, mark]
        distAndMark = []

        # for each training data
        for j in range(len(dfTrain)):
            thisTrainData = dfTrain[j]

            # calculate distance from the test data
            thisDistSquare = 0
            for l in range(len(dfTrain[0])):
                if l == targetIndex: continue
                thisDistSquare = thisDistSquare + pow(thisTestData[l] - thisTrainData[l], 2)

            # add to distAndMark
            distAndMark.append([math.sqrt(thisDistSquare), thisTrainData[targetIndex]])

        # sort distAndMark array
        distAndMark = sorted(distAndMark, key=lambda x:x[0], reverse=False)

        # count the vote for each class (using weight = len(dfTrain)/trainCount)
        vote = {} # vote result for each class: [class targetVals[j], vote score of targetVals[j]]
        for j in range(len(classCount)): vote[targetVals[j]] = 0 # initialize dictionary vote
        for j in range(k): # count the vote using k nearest neighbors
            thisMark = distAndMark[j][1] # mark of this 'neighbor'
            vote[thisMark] = vote[thisMark] + len(dfTrain) / classCount[thisMark]

        # find max vote item
        largestVoteVal = -1 # number of votes of largest voted target value
        largestVoteTargetVal = -1 # largest voted target value

        # key: class targetVals[j], value: vote score of targetVals[j]
        for key in vote.keys():
            value = vote[key]
            
            if value > largestVoteVal:
                largestVoteVal = value
                largestVoteTargetVal = key

        # append the vote result (=prediction) to result array
        result.append(largestVoteTargetVal)
        resultT.append([largestVoteTargetVal])

    # add vote result value
    # https://rfriend.tistory.com/352
    dfTest = np.column_stack([dfTest, resultT])

    # print data as 2d or 3d space
    if len(dfTest[0]) == 3: # 2 except for target col
        printDataAsSpace(2, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'target']))
    elif len(dfTest[0]) == 4: # 3 except for target col
        printDataAsSpace(3, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'pca2', 'target']))
            
    # return the result array
    return result

# make PCA from training data
targetCol = 'requester_received_pizza'
tfCols = ['post_was_edited', 'requester_received_pizza']
textCols = ['giver_username_if_known', 'request_id', 'request_text', 'request_text_edit_aware',
                'request_title', 'requester_subreddits_at_request', 'requester_user_flair',
                'requester_username']
exceptCols = ["number_of_downvotes_of_request_at_retrieval",
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
              "unix_timestamp_of_request_utc"] # list of columns not used

# get PCA (components and explained variances) for training data
(df_pca_train, comp, exva, mean, targetCol) = makePCA('train.json', 3, True, targetCol, tfCols, textCols, exceptCols,
                                                      None, None, None)

# remove target column from comp and mean
comp = np.delete(comp, [targetCol], 1)
mean = np.delete(mean, [targetCol], 0)

# get PCA (components and explained variances) for test data
(df_pca_test, noUse0, noUse1, noUse2, noUse3) = makePCA('test.json', 3, False, targetCol, tfCols, textCols, exceptCols,
                                                        comp, exva, mean)

kNN(df_pca_train, df_pca_test, 'target', 3, 10)

# test
for i in range(len(json_data)):
    if y[i] <= 1:
        result += str(x[i]) + ',' + str(0) + '\n'
    else:
        result += str(x[i]) + ',' + str(1) + '\n'

f = open('result.csv', 'w')
f.write(result)
f.close()

# 향후계획
# train 시의 PCA와 test 시의 PCA를 일치시키기 (PCA에서 특정 column을 제거하는 방법
#     / 하나의 PCA로 train+test를 한번에 학습시키는 방법 필요) [FINISHED]
# decision tree 모델 도입
# categorical data(textCols 중 값의 종류가 일정개수 이하)를 one-hot으로 처리하여 숫자로 변환
