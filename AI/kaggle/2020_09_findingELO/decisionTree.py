import numpy as np
from sklearn import tree
from sklearn.tree import export_text
import pandas as pd
import printData as _PD

# create Decision Tree using DataFrame
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# dataSetDF    : dataframe to create Decision Tree
# targetCol    : index of target column
# displayChart : display as chart?
# DT_maxDepth  : max depth of decision tree
# DT_criterion : 'gini' or 'entropy'
# DT_splitter  : 'best' or 'random'
# numericRange : convert to mean of each range when target column is numeric
#                (available only when target column is numeric)
#                for example, if [100, 200, 300, 450], convert 115, 150, 190 to mean of [100, 200] = 150
def createDTfromDF(dataSetDF, targetCol, displayChart, DT_maxDepth, DT_criterion, DT_splitter, numericRange):
    print('')
    print('+=============================+')
    print('|  Function : createDTfromDF  |')
    print('+=============================+')

    cols = len(dataSetDF.columns) # number of columns

    # designate index for input data and output(target) data
    inputIndex = []
    for i in range(cols):
        if i != targetCol: inputIndex.append(i)

    outputIndex = [targetCol]

    # extract input and output data of dataSetDF
    inputData = np.array(dataSetDF.iloc[:, inputIndex])
    outputData = np.array(dataSetDF.iloc[:, outputIndex]).flatten()

    # print output data
    print('\n<<< [3-pre0] output data (first 100 elements) >>>')
    print(outputData[:min(len(outputData), 100)])

    # convert using numericRange
    if numericRange != None:
        for i in range(len(outputData)):
            for j in range(len(numericRange)-1):
                if outputData[i] >= numericRange[j] and outputData[i] < numericRange[j+1]:
                    outputData[i] = (numericRange[j] + numericRange[j+1]) / 2

    # print output data
    print('\n<<< [3-pre1] output data (first 100 elements) >>>')
    print(outputData[:min(len(outputData), 100)])

    # type setting for data
    inputData = inputData.astype('float')
    outputData = outputData.astype('str')

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
            _PD.printDataAsSpace(2, pd.DataFrame(dataSetDF, columns=['pca0', 'pca1', 'target']), title)
        elif cols == 4: # 3 except for target col
            _PD.printDataAsSpace(3, pd.DataFrame(dataSetDF, columns=['pca0', 'pca1', 'pca2', 'target']), title)
    
    return DT

# predict using Decision Tree
# df_pca_test  : dataFrame for test data
# DT           : decision tree
# displayChart : display as chart?
# DT_maxDepth  : max depth of decision tree
# DT_criterion : 'gini' or 'entropy'
# DT_splitter  : 'best' or 'random'
def predictDT(df_pca_test, DT, displayChart, DT_maxDepth, DT_criterion, DT_splitter):
    print('')
    print('+========================+')
    print('|  Function : predictDT  |')
    print('+========================+')

    # get prediction
    DTresult = DT.predict(df_pca_test)

    # print prediction
    print('\n<<< [4-0] DTresult (first 100 elements) >>>')
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
            _PD.printDataAsSpace(2, pd.DataFrame(df_pca_test_new, columns=['pca0', 'pca1', 'target']), title)
        elif len(df_pca_array[0]) == 4: # 3 except for target col
            _PD.printDataAsSpace(3, pd.DataFrame(df_pca_test_new, columns=['pca0', 'pca1', 'pca2', 'target']), title)

    return DTresult
