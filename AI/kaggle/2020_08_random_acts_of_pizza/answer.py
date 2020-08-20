import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab
import json

# to make decision tree
# https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

# for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# fn     : file name
# n_cols : number of columns(components) of PCA
# isTrain: training(True) or not(False)
def makePCA(fn, n_cols, isTrain):

    # open and show plt data
    jf = open(fn, 'r')
    json_loaded = json.load(jf)
    json_data = pd.DataFrame(json_loaded)

    # True to 1, False to 0, and decimal to 0
    # https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false
    tfCols = ['post_was_edited', 'requester_received_pizza']
    tfColsIndex = [-1, -1] # column indices of tfCols

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
    if isTrain == True: targetCol = 'requester_received_pizza' # target column name
    textCols = ['giver_username_if_known', 'request_id', 'request_text', 'request_text_edit_aware',
                'request_title', 'requester_subreddits_at_request', 'requester_user_flair',
                'requester_username']
    
    dataPart = [] # data columns
    if isTrain == True: targetPart = [] # target column
    extractCols = [] # extracted columns = dataPart + targetPart

    for col in dataCols:
        dataPartAdded = False

        # accept columns only if not all values are the same (then not meaningful)
        # so check if max(col) > min(col)
        
        # not in targetCol and all of values are numeric -> dataPart
        if isTrain == True: # train mode -> targetCol exists
            if col != targetCol and not col in textCols:
                if max(json_data[col]) > min(json_data[col]):
                    dataPart.append(col)
                    extractCols.append(col)
                    dataPartAdded = True
        else: # test mode -> targetCol does not exist
            if not col in textCols:
                if max(json_data[col]) > min(json_data[col]):
                    dataPart.append(col)
                    extractCols.append(col)
                    dataPartAdded = True

        # equal to targetCol
        if isTrain == True and dataPartAdded == False:
            if col == targetCol:
                targetPart.append(col)
                extractCols.append(col)

    # bind the data and target
    if isTrain == True: dataSet = {'data':json_data[dataPart], 'target':json_data[targetPart]}
    else: dataSet = {'data':json_data[dataPart]}
    
    dataSetDF = json_data[extractCols]
    dataSetDF = dataSetDF.astype(float) # change dataSetDF into float type

    # display correlation
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    print('\n<<< [1] dataSetDF >>>')
    print(dataSetDF)

    df = dataSetDF.corr() # get correlation
    seab.clustermap(df,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # PCA
    # https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
    scaled = StandardScaler().fit_transform(dataSetDF) # to standard normal distribution
    pca = PCA(n_components=n_cols)

    # get PCA transformed data
    pca.fit(scaled)
    scaledPCA = pca.transform(scaled)

    print('\n<<< [2] scaledPCA.shape >>>\n' + str(scaledPCA.shape))
    print('\n<<< [3] scaledPCA.data.shape >>>\n' + str(scaledPCA.data.shape))

    print('\n<<< [4] scaledPCA >>>')
    print(scaledPCA)

    # convert to numpy array
    json_data = np.array(json_data)

    # name each column for PCA transformed data
    pca_cols = []
    for i in range(n_cols): pca_cols.append('pca' + str(i))
    df_pca = pd.DataFrame(scaledPCA, columns=pca_cols)
    if isTrain == True: df_pca['target'] = dataSetDF['requester_received_pizza']

    print('\n<<< [5] df_pca >>>')
    print(df_pca)

    df_pcaCorr = df_pca.corr()
    seab.clustermap(df_pcaCorr,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # immediately return the pca if testing
    if isTrain == False: return df_pca

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

    # return
    return df_pca

# make PCA from training data
df_pca = makePCA('train.json', 3, True)

# test
for i in range(len(json_data)):
    if y[i] <= 1:
        result += str(x[i]) + ',' + str(0) + '\n'
    else:
        result += str(x[i]) + ',' + str(1) + '\n'

f = open('result.csv', 'w')
f.write(result)
f.close()
