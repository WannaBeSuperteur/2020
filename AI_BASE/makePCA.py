import makeDF as _MDF
import printData as _PD
import matplotlib.pyplot as plt
import seaborn as seab
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# fn           : file name
# validExcept  : except for these columns for training data for validation
# rows         : rows to use, in 1d array form (if None, use all rows)
# ftype        : file type (json, csv, txt)
# fcols        : columns (if ftype is 'txt' and fcols is None, use default name)
# n_cols       : number of columns(components) of PCA
# isTrain      : training(True) or not(False)
# target       : target column if isTrain is True
# exceptCols   : using the columns except for them
# comp         : components of PCA used
# exva         : explained variances of PCA used
# mean         : mean of PCA used
# useLog       : using log for making dataFrame
# logConstant  : x -> log2(x + logConstant)
def makePCA(fn, validExcept, rows, ftype, fcols, n_cols, isTrain, target, exceptCols, comp, exva, mean, exceptTargetForPCA, useLog, logConstant):
    print('')
    print('+=======================+')
    print('|  Function : makePCA   |')
    print('+=======================+')

    # get dataFrame
    (dataSetDF, targetCol) = _MDF.makeDataFrame(fn, validExcept, rows, ftype, fcols, isTrain, target, exceptCols, useLog, logConstant)
    DFtoFindPCA = dataSetDF # dataFrame to find PCA

    # remove target column when exceptTargetForPCA is True
    if exceptTargetForPCA == True:
        newDataSetDF = dataSetDF.drop([target], axis='columns')

        # print newDataSetDF
        print('\n<<< [8] newDataSetDF.columns >>>')
        print(newDataSetDF.columns)
        print('\n<<< [9] newDataSetDF >>>')
        print(newDataSetDF)

        DFtoFindPCA = newDataSetDF
    
    # display correlation
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    df = DFtoFindPCA.corr() # get correlation
    seab.clustermap(df,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # to standard normal distribution
    scaled = StandardScaler().fit_transform(DFtoFindPCA)
    
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
    print('\n<<< [10] pca.components_ >>>\n' + str(comp))
    print('\n<<< [11] pca.explained_variance_ >>>\n' + str(exva))
    print('\n<<< [12] pca.mean_ >>>\n' + str(mean))

    # create PCA using comp and exva
    if initializePCA == False:
        pca = PCA(n_components=n_cols)
        pca.components_ = comp
        pca.explained_variance_ = exva
        pca.mean_ = mean
    
    # apply PCA to the data
    scaledPCA = pca.transform(scaled)

    print('\n<<< [13] scaledPCA.shape >>>\n' + str(scaledPCA.shape))
    print('\n<<< [14] scaledPCA.data.shape >>>\n' + str(scaledPCA.data.shape))

    print('\n<<< [15] scaledPCA >>>')
    print(scaledPCA)

    # for training data
    # (ID : ~original train data) -> (ID : ~except for validation data)
    if isTrain == True:
        print('\n<<< [15-1] dataSetDF[target] before >>>')
        print(dataSetDF[target])

        # dataFrame -> list -> dataFrame
        targetList = list(dataSetDF[target])
        targetListCopy = []
        for i in range(len(targetList)): targetListCopy.append(targetList[i])
        targetDF = pd.DataFrame(targetListCopy)

        print('\n<<< [15-2] dataSetDF[target] after : targetDF >>>')
        print(targetDF)

    # name each column for PCA transformed data
    pca_cols = []
    for i in range(n_cols): pca_cols.append('pca' + str(i))
    df_pca = pd.DataFrame(scaledPCA, columns=pca_cols)
    if isTrain == True: df_pca['target'] = targetDF

    print('\n<<< [16] df_pca >>>')
    print(df_pca)

    df_pcaCorr = df_pca.corr()
    seab.clustermap(df_pcaCorr,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # immediately return the pca if testing
    if isTrain == False:
        print('')
        print('+=======================+')
        print('|    Exit : makePCA     |')
        print('+=======================+')

        return (df_pca, comp, exva, mean, targetCol)

    # print data as 2d or 3d space (run only on training data)
    _PD.printDataAsSpace(n_cols, df_pca, '(PCA) training data')

    print('')
    print('+=======================+')
    print('|    Exit : makePCA     |')
    print('+=======================+')

    return (df_pca, comp, exva, mean, targetCol)
