import math
import random
import os
import numpy as np
import readData as RD
import main as main

# extract from array
def extractColumns(array, columns):
    
    toReturn = []
    for i in range(len(columns)): toReturn.append(array[columns[i]])
    return toReturn

# extract from array except for columns
def extractExceptForColumns(array, columns):
    toReturn = []
    for i in range(len(array)): toReturn.append(array[i])
    for i in range(len(columns)-1, -1, -1): toReturn.pop(columns[i])
    return toReturn

# split array with specified conditions, and return colValues
# fn      : array file name
# columns : split with values for these columns
#           example: [0, 1, 4] if using 0th, 1st, and 4th column of the array
def splitArray(fn, columns):

    # load array
    originalArray = RD.loadArray(fn)

    # duplicated-removed arrays of column-value info
    # example: array   = [[10, 15, 2], [10, 15, 3], [10, 8, 4], [16, 8, 5], [10, 8, 6],
    #                     [10, 8, 7], [10, 15, 8], [10, 15, 9], [16, 8, 10], [10, 15, 11]]
    #          columns = [0, 1]
    #       -> aBCV    = [
    #                     [[10, 15], [[2], [3], [8], [9], [11]]],
    #                     [[10, 8], [[4], [6], [7]]],
    #                     [[16, 8], [[5], [10]]]
    #                    ]
    #       -> return    [[10, 15], [10, 8], [16, 8]]
    #
    # aBCV = arrayByColValues
    arrayByColValues = []

    # make array like [[10, 15], [10, 8], [16, 8]]
    colValues = []
    for i in range(len(originalArray)):

        toAppend = [] # like [10, 15]
        for j in range(len(columns)):
            toAppend.append(originalArray[i][columns[j]])

        if toAppend not in colValues: # except for duplicate
            colValues.append(toAppend)

    # split data and save files using colValues
    # after, arrayByColValues may be like [[[10, 15], []], [[10, 8], []], [[16, 8], []]]
    for i in range(len(colValues)):
        arrayByColValues.append([colValues[i], []])

    # create splitted array (arrayByColValues)
    for i in range(len(originalArray)):
        if i % 100 == 0: print(str(i) + ' / ' + str(len(originalArray)))

        # extract values corresponding to 'columns' of original array
        # for example, columns = [0, 1] and originalArray[i] = [10, 15, 2] -> [10, 15]
        extract = extractColumns(originalArray[i], columns)

        # find matching values (with extracted values) from colValues
        for j in range(len(colValues)):

            # [10, 15] for example
            thisColValues = colValues[j]

            # skip non-matching columns
            if extract != thisColValues: continue

            # [2] for example
            valuesExceptExtracted = extractExceptForColumns(originalArray[i], columns)

            # [[10, 15], []] -> [[10, 15], [[2]]] for example
            arrayByColValues[j][1].append(valuesExceptExtracted)

    # save as files
    for i in range(len(colValues)):
        print(i)
        RD.saveArray(fn[:len(fn)-4] + '_sub_' + str(i) + '.txt', arrayByColValues[i][1])

    # return splitted array
    return colValues

# use deep learning for each piece of data
# configFn  : configuration file name
# colValues : colValues in splitArray function

# needed arguments: fcolsTrain, fcolsTest, exceptCols, usePCA, validationExceptCols, exceptTargetForPCA,
#                   useLog, kNN_useAverage, kNN_useCaseWeight, kNN_weight, DT_numericRange, exceptColsForMethod2,
#                   XG_info and DL_normalizeTarget
#                   (logical or array arguments)
def applyAlgorithmToSplittedArray(fcolsTrain, fcolsTest, exceptCols, usePCA, validationExceptCols,
                                  exceptTargetForPCA, useLog, kNN_useAverage, kNN_useCaseWeight,
                                  kNN_weight, DT_numericRange, exceptColsForMethod2, XG_info, DL_normalizeTarget,
                                  configFn, colValues):
    
    ### read configuration file ###
    
    f = open(configFn, 'r')
    fl = f.readlines()
    f.close()
    for i in range(len(fl)): fl[i] = fl[i].split('\n')[0]

    ### init configuration values as None ###
    
    # basic configuration
    finalResult = None 
    trainName = None # except for '_sub_X.txt'
    testName = None # except for '_sub_X.txt'
    ftype = None
    targetColName = None

    # important configuration
    globalValidationRate = None
    method = None
    validationCol = None

    # PCA and log
    PCAdimen = None
    logConstant = None

    # for kNN (method 0)
    kNN_k = None

    # for Decision Tree (method 1)
    DT_maxDepth = None
    DT_criterion = None
    DT_splitter = None

    ### extract configuration ###
    
    for i in range(len(fl)):
        configSplit = fl[i].split(' ') # split

        if configSplit[0] == 'finalResult': finalResult = configSplit[1]
        elif configSplit[0] == 'trainName': trainName = configSplit[1] # except for '_sub_X.txt'
        elif configSplit[0] == 'testName': testName = configSplit[1] # except for '_sub_X.txt'
        elif configSplit[0] == 'ftype': ftype = configSplit[1]
        elif configSplit[0] == 'targetColName': targetColName = configSplit[1]
        elif configSplit[0] == 'globalValidationRate': globalValidationRate = float(configSplit[1])
        elif configSplit[0] == 'method': method = int(configSplit[1])
        elif configSplit[0] == 'validationCol': validationCol = int(configSplit[1])
        elif configSplit[0] == 'PCAdimen': PCAdimen = int(configSplit[1])
        elif configSplit[0] == 'logConstant': logConstant = int(configSplit[1])
        elif configSplit[0] == 'kNN_k': kNN_k = int(configSplit[1])
        elif configSplit[0] == 'DT_maxDepth': DT_maxDepth = int(configSplit[1])
        elif configSplit[0] == 'DT_criterion': DT_criterion = configSplit[1]
        elif configSplit[0] == 'DT_splitter': DT_splitter = configSplit[1]

    ### execute algorithm ###

    # for each file trainName_sub_0.txt, trainName_sub_1.txt, ...
    for i in range(len(colValues)):

        trainName_i = trainName + '_sub_' + str(i) + '.txt' # training data file name (also used for validation)
        testName_i = testName + '_sub_' + str(i) + '.txt' # test data file name

        ### copy array (to prevent modification of original array) ###
        
        fcolsTrain_ = []
        fcolsTest_ = []
        exceptCols_ = []
        validationExceptCols_ = []
        kNN_weight_ = []
        DT_numericRange_ = []
        exceptColsForMethod2_ = []
        XG_info_ = []

        for j in range(len(fcolsTrain)): fcolsTrain_.append(fcolsTrain[j])
        for j in range(len(fcolsTest)): fcolsTest_.append(fcolsTest[j])
        for j in range(len(exceptCols)): exceptCols_.append(exceptCols[j])
        for j in range(len(validationExceptCols)): validationExceptCols_.append(validationExceptCols[j])
        for j in range(len(kNN_weight)): kNN_weight_.append(kNN_weight[j])
        for j in range(len(DT_numericRange)): DT_numericRange_.append(DT_numericRange[j])
        for j in range(len(exceptColsForMethod2)): exceptColsForMethod2_.append(exceptColsForMethod2[j])
        for j in range(len(XG_info)): XG_info_.append(XG_info[j])

        ### execute ###
        
        main.executeAlgorithm(finalResult, trainName_i, testName_i, ftype, fcolsTrain_, fcolsTest_, targetColName, exceptCols_, # basic configuration
                         globalValidationRate, method, usePCA, validationExceptCols_, validationCol, # important configuration
                         PCAdimen, exceptTargetForPCA, useLog, logConstant, # PCA and log
                         kNN_k, kNN_useAverage, kNN_useCaseWeight, kNN_weight_, # for kNN (method 0)
                         DT_maxDepth, DT_criterion, DT_splitter, DT_numericRange_, # for Decision Tree (method 1)
                         exceptColsForMethod2_, # for method 2
                         XG_info_, # xgBoost (for method 3 and 4)
                         DL_normalizeTarget) # for Deep Learning (method 5 and 6)

        ### change file names ###

        # change name of result_valid.txt
        if globalValidationRate > 0: os.rename('result_valid.txt', 'result_split_valid_' + str(i) + '.txt')

        # change name of result.csv
        os.rename('result.csv', 'result_split_' + str(i) + '.csv')

    # return True if globalValidation > 0, else False
    return globalValidationRate > 0

# merge test result using IDs of original training data file (test only, not for validation)
# original : original test data file name (without '.txt')
# IDcol    : ID column, for example 0
# files    : number of files created by splitArray function
def mergeTestResult(original, IDcol, files):
    merged = []

    for i in range(files):

        # load test result array from result file
        result = RD.loadArray('result_split_' + str(i) + '.csv', ',')
        result = result[1:] # remove first value with column title

        # read corresponding ID
        test = RD.loadArray(original + '_sub_' + str(i) + '.txt')
        testIDs_ = np.array(test)[:, IDcol]

        # testIDs_ = [0, 1, 2, ...] -> testIDs = [[0], [1], [2], ...]
        testIDs = []
        for j in range(len(testIDs_)): testIDs.append([testIDs_[j]])

        # merge result and test array
        IDsAndResult = np.concatenate([np.array(testIDs), np.array(result)], axis=1)

        print('\n <<< IDs-Result for file ' + str(i) + ' >>>')
        print(np.array(IDsAndResult))

        # append this to array 'merged'
        for j in range(len(IDsAndResult)): merged.append([IDsAndResult[j][0], IDsAndResult[j][1]])

    # sort the result array 'merged'
    merged = sorted(merged, key=lambda x:x[0])

    # write to file
    RD.saveArray('result_split_final.txt', merged)

if __name__ == '__main__':
    cols = [1, 2, 3]
    colValues_train = splitArray('data_trainPgn.txt', cols)
    colValues_test = splitArray('data_testPgn.txt', cols)

    # settings
    # fcolsTrain and fcolsTest: except for columns in 'cols' because it is not used
    fcolsTrain = ['id', 'welo', 'belo', 'score0', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7']
    fcolsTest = ['id', 'score0', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7']
    targetColName = 'belo'
    exceptCols = ['id', 'welo'] # NO need to add columns in 'cols' when using splitted data

    usePCA = False
    validationExceptCols = ['welo', 'belo'] # NO need to add columns in 'cols' when using splitted data

    exceptTargetForPCA = True # except target column for PCA
    useLog = False # using log for numeric data columns

    kNN_useAverage = True # use average voting for kNN
    kNN_useCaseWeight = False # use case weight (weight by number of cases from training data) for kNN
    kNN_weight = [1, 1, 1, 1.5, 1.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] # weight for kNN (for each test data column)

    DT_numericRange = range(1600, 2500, 30)

    exceptColsForMethod2 = ['id', 'welo'] # list of columns not used for method 2

    XG_epochs = 100 # xgboost training epochs
    XG_boostRound = 10000 # num_boost_round for xgb.train
    XG_earlyStoppingRounds = 10 # early_stopping_rounds for xgb.train
    XG_foldCount = 5 # XG_foldCount = N -> (N-1)/N of whole dataset is for training, 1/N is for test
    XG_rateOf1s = 0.15 # rate of 1's (using this, portion of 1 should be XG_rateOf1s)
    XG_info = [XG_epochs, XG_boostRound, XG_earlyStoppingRounds, XG_foldCount, XG_rateOf1s]

    DL_normalizeTarget = False # normalize training output value? (False because automatically normalized)

    configFn = 'config_split.txt' # configuration file name
    colValues = colValues_train # each splitted type: [[10, 15], [10, 8], [16, 8]]

    # apply algorithm
    isValid = applyAlgorithmToSplittedArray(fcolsTrain, fcolsTest, exceptCols, usePCA, validationExceptCols,
                                          exceptTargetForPCA, useLog, kNN_useAverage, kNN_useCaseWeight,
                                          kNN_weight, DT_numericRange, exceptColsForMethod2, XG_info, DL_normalizeTarget,
                                          configFn, colValues)

    # merge and save test result
    if isValid == False: mergeTestResult('data_testPgn', 0, 3)
