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
from sklearn.tree import export_text

# print data point as 2d or 3d space
# data should be PCA-ed data
# n_cols : number of columns (not include target)
# df_pca : dataFrame
# title  : title of displayed chart
def printDataAsSpace(n_cols, df_pca, title):

    # set markers
    markers = ['^', 'o']

    # plot if the value of n_cols is 2
    # https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
    if n_cols == 2:

        # add each point
        for i, marker in enumerate(markers):
            x_axis_data = df_pca[df_pca['target']==i]['pca0']
            y_axis_data = df_pca[df_pca['target']==i]['pca1']
            plt.scatter(x_axis_data, y_axis_data, s=1, marker=marker, label=df_pca['target'][i])

        # set labels and show
        plt.title(title)
        plt.legend()
        plt.xlabel('pca0')
        plt.ylabel('pca1')
        plt.show()

    # plot in the space, if the value of n_cols is 3
    # https://python-graph-gallery.com/372-3d-pca-result/
    elif n_cols == 3:

        fig = plt.figure()
        fig.suptitle(title)
        
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca['pca0'], df_pca['pca1'], df_pca['pca2'], c=df_pca['target'], s=1)

        # set labels and show
        ax.set_xlabel('pca0')
        ax.set_ylabel('pca1')
        ax.set_zlabel('pca2')
        plt.show()

    else: print('n_cols should be 2 or 3 to print data as 2d/3d space')

# fn        : file name
# isTrain   : training(True) or not(False)
# target    : target column if isTrain is True
# tfCols    : columns that contains True or False values
# textCols  : columns that contains text values
# exceptCols: using the columns except for them
def makeDataFrame(fn, isTrain, target, tfCols, textCols, exceptCols):
    print('\n ######## makeDataFrame function ########')
    
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
    print('\n<<< [0] json_data.shape >>>\n' + str(json_data.shape))

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

    # return dataFrame
    print('\n<<< [2] dataSetDF >>>')
    print(dataSetDF)

    return (dataSetDF, targetCol)

# create Decision Tree using DataFrame
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# dataSetDF    : dataframe to create Decision Tree
# targetCol    : index of target column
# displayChart : display as chart?
# DT_maxDepth  : max depth of decision tree
# DT_criterion : 'gini' or 'entropy'
# DT_splitter  : 'best' or 'random'
def createDTfromDF(dataSetDF, targetCol, displayChart, DT_maxDepth, DT_criterion, DT_splitter):
    print('\n ######## createDTfromDF function ########')

    cols = len(dataSetDF.columns) # number of columns

    # designate index for input data and output(target) data
    inputIndex = []
    for i in range(cols):
        if i != targetCol: inputIndex.append(i)

    outputIndex = [targetCol]

    # extract input and output data of dataSetDF
    inputData = np.array(dataSetDF.iloc[:, inputIndex])
    outputData = np.array(dataSetDF.iloc[:, outputIndex]).flatten()

    # create Decision Tree using input and output data
    DT = tree.DecisionTreeClassifier(max_depth=DT_maxDepth, criterion=DT_criterion, splitter=DT_splitter)
    DT = DT.fit(inputData, outputData)

    print('\n<<< [3] DT (Decision Tree) >>>')
    r = export_text(DT, feature_names=None)
    print(r)

    # display as chart
    if displayChart == True:
        title = ('(Decision Tree) training data / maxDepth='
                 + str(DT_maxDepth) + ', criterion=' + DT_criterion + ', splitter=' + DT_splitter)
        
        # print data as 2d or 3d space
        if cols == 3: # 2 except for target col
            printDataAsSpace(2, pd.DataFrame(dataSetDF, columns=['pca0', 'pca1', 'target']), title)
        elif cols == 4: # 3 except for target col
            printDataAsSpace(3, pd.DataFrame(dataSetDF, columns=['pca0', 'pca1', 'pca2', 'target']), title)
    
    return DT

# predict using Decision Tree
# df_pca_test  : dataFrame for test data
# DT           : decision tree
# displayChart : display as chart?
# DT_maxDepth  : max depth of decision tree
# DT_criterion : 'gini' or 'entropy'
# DT_splitter  : 'best' or 'random'
def predictDF(df_pca_test, DT, displayChart, DT_maxDepth, DT_criterion, DT_splitter):
    print('\n ######## predictDF function ########')

    # get prediction
    DTresult = DT.predict(df_pca_test)

    # print prediction
    print('\n<<< [4] DTresult (first 100 elements) >>>')
    print(DTresult[:min(len(DTresult), 100)])

    # display as chart
    if displayChart == True:
        title = ('(Decision Tree) test data prediction / maxDepth='
                 + str(DT_maxDepth) + ', criterion=' + DT_criterion + ', splitter=' + DT_splitter)

        # add result to df_pca_test
        colsSaved = df_pca_test.columns
        colsSaved = colsSaved.append(pd.Index(['target']))
        df_pca_array = np.array(df_pca_test)
        df_pca_array = np.column_stack([df_pca_array, DTresult])
        df_pca_test_new = pd.DataFrame(df_pca_array, columns=colsSaved)

        print('\n<<< [5] df_pca_array >>>')
        print(df_pca_array)

        print('\n<<< [6] df_pca_test_new >>>')
        print(df_pca_test_new)

        print('\n<<< [7] colsSaved >>>')
        print(colsSaved)
        
        # print data as 2d or 3d space
        if len(df_pca_array[0]) == 3: # 2 except for target col
            printDataAsSpace(2, pd.DataFrame(df_pca_test_new, columns=['pca0', 'pca1', 'target']), title)
        elif len(df_pca_array[0]) == 4: # 3 except for target col
            printDataAsSpace(3, pd.DataFrame(df_pca_test_new, columns=['pca0', 'pca1', 'pca2', 'target']), title)

    return DTresult

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

    # get dataFrame
    (dataSetDF, targetCol) = makeDataFrame(fn, isTrain, target, tfCols, textCols, exceptCols)
    
    # display correlation
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
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
    print('\n<<< [8] pca.components_ >>>\n' + str(comp))
    print('\n<<< [9] pca.explained_variance_ >>>\n' + str(exva))
    print('\n<<< [10] pca.mean_ >>>\n' + str(mean))

    # create PCA using comp and exva
    if initializePCA == False:
        pca = PCA(n_components=n_cols)
        pca.components_ = comp
        pca.explained_variance_ = exva
        pca.mean_ = mean
    
    # apply PCA to the data
    scaledPCA = pca.transform(scaled)

    print('\n<<< [11] scaledPCA.shape >>>\n' + str(scaledPCA.shape))
    print('\n<<< [12] scaledPCA.data.shape >>>\n' + str(scaledPCA.data.shape))

    print('\n<<< [13] scaledPCA >>>')
    print(scaledPCA)

    # name each column for PCA transformed data
    pca_cols = []
    for i in range(n_cols): pca_cols.append('pca' + str(i))
    df_pca = pd.DataFrame(scaledPCA, columns=pca_cols)
    if isTrain == True: df_pca['target'] = dataSetDF[target]

    print('\n<<< [14] df_pca >>>')
    print(df_pca)

    df_pcaCorr = df_pca.corr()
    seab.clustermap(df_pcaCorr,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # immediately return the pca if testing
    if isTrain == False: return (df_pca, comp, exva, mean, targetCol)

    # print data as 2d or 3d space (run only on training data)
    printDataAsSpace(n_cols, df_pca, '(PCA) training data')

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

    # convert to numpy array
    dfTrain = np.array(dfTrain)
    dfTest = np.array(dfTest)
    print('\n<<< [15] dfTrain >>>')
    print(dfTrain)
    print('\n<<< [16] dfTest >>>')
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

    # display as chart
    title = '(kNN) test data prediction'

    # print data as 2d or 3d space
    if len(dfTest[0]) == 3: # 2 except for target col
        printDataAsSpace(2, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'target']), title)
    elif len(dfTest[0]) == 4: # 3 except for target col
        printDataAsSpace(3, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'pca2', 'target']), title)
            
    # return the result array
    return result

if __name__ == '__main__':

    # meta info
    trainName = 'train.json'
    testName = 'test.json'

    # make PCA from training data
    PCAdimen = 3 # dimension of PCA
    idCol = 'request_id'
    targetColName = 'requester_received_pizza'
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

    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    useDecisionTree = False
    DT_maxDepth = 10 # max depth of decision tree
    DT_criterion = 'gini' # 'gini' or 'entropy'
    DT_splitter = 'best' # 'best' or 'random'

    # get PCA (components and explained variances) for training data
    (df_pca_train, comp, exva, mean, targetCol) = makePCA(trainName, PCAdimen, True, targetColName, tfCols, textCols, exceptCols,
                                                          None, None, None)

    # remove target column from comp and mean
    comp = np.delete(comp, [targetCol], 1)
    mean = np.delete(mean, [targetCol], 0)

    # get PCA (components and explained variances) for test data
    (df_pca_test, noUse0, noUse1, noUse2, noUse3) = makePCA(testName, PCAdimen, False, None, tfCols, textCols, exceptCols,
                                                            comp, exva, mean)

    # use decision tree
    if useDecisionTree == True:
        
        # make decision tree
        DT = createDTfromDF(df_pca_train, PCAdimen, True, DT_maxDepth, DT_criterion, DT_splitter)

        # predict test data using decision tree
        finalResult = predictDF(df_pca_test, DT, True, DT_maxDepth, DT_criterion, DT_splitter)

    # do not use decision tree
    else:

        # k-NN of test data
        finalResult = kNN(df_pca_train, df_pca_test, 'target', PCAdimen, 10)

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

# 향후계획
# Decision Tree의 하위노드 개수, truncate되는 단계를 조정, best/random 설정 [ING]
# -> https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# 학습단계에서 target column을 제외하고 PCA변환하는 것을 고려
# 전체 열에 대한 단일 Decision Tree를 이용하여 결정
# textCols에서 특정 텍스트의 등장여부를 열로 추가하여 PCA 분석에 추가하는 것을 고려
