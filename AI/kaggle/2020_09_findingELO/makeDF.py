import pandas as pd
import numpy as np
import readData as RD

# get index of the column from df_array, using original fcols
# the column of colName is also removed from originalFcols 
def getIndex(colName, originalFcols):
    for i in range(len(originalFcols)):
        if originalFcols[i] == colName:
            originalFcols.remove(colName)
            return i
    return None

# fn           : file name
# validExcept  : except for these columns for training data for validation
# rows         : rows to use, in 1d array form (if None, use all rows)
# ftype        : file type (json, csv, txt)
# fcols        : columns (if ftype is 'txt' and fcols is None, use default name)
# isTrain      : training(True) or not(False)
# target       : target column name for training data (removed for test data)
# exceptCols   : using the columns except for them
# useLog       : using log for making dataFrame
# logConstant  : x -> log2(x + logConstant)
def makeDataFrame(fn, validExcept, rows, ftype, fcols, isTrain, target, exceptCols, useLog, logConstant):
    print('')
    print('+============================+')
    print('|  Function : makeDataFrame  |')
    print('+============================+')

    # copy fcols (original fcols)
    originalFcols = []
    for i in range(len(fcols)): originalFcols.append(fcols[i])
    
    # open and show plt data
    if ftype == 'json': # type is 'json'
        jf = open(fn, 'r')
        df_loaded = json.load(jf)
        df_data = pd.DataFrame(df_loaded)

        # NOT TESTED FOR THIS CASE - SO THERE MAY BE SOME ERRORS
        
    elif ftype == 'csv': # type is 'csv'
        df_data = pd.read_csv(fn)

        # NOT TESTED FOR THIS CASE - SO THERE MAY BE SOME ERRORS
        
    elif ftype == 'txt': # type is 'txt'
        df_array = RD.loadArray(fn)

        print('\n<<< [before] fcols >>>')
        print(fcols)
        print('\n<<< [before] data array [0:5] >>>')
        print(df_array[:5])

        # remove columns indexed by elements of validExcept
        # (used in training but not used in validation)
        if validExcept != None:
            for i in range(len(validExcept)):
                try:
                    fcols.remove(validExcept[i])
                except:
                    print('validExcept remove error (0) : ' + str(validExcept[i]))
                try:
                    df_array = np.delete(df_array, getIndex(validExcept[i], originalFcols), 1)
                except:
                    print('validExcept remove error (1) : ' + str(validExcept[i]))

        # remove target column for test data
        if isTrain == False:
            try:
                fcols.remove(target)
            except:
                print('target      remove error (0) : ' + str(target))
            try:
                df_array = np.delete(df_array, getIndex(target, originalFcols), 1)
            except:
                print('target      remove error (1) : ' + str(target))

        # remove except column from both fcols and df_array
        for exceptCol in exceptCols:
            try:
                fcols.remove(exceptCol)
            except:
                print('exceptCol   remove error (0) : ' + str(exceptCol))
            try:
                df_array = np.delete(df_array, getIndex(exceptCol, originalFcols), 1)
            except:
                print('exceptCol   remove error (1) : ' + str(exceptCol))

        print('\n<<< [after] fcols >>>')
        print(fcols)
        print('\n<<< [after] data array [0:5] >>>')
        print(df_array[:5])

        # make dataframe using df_array
        if fcols != None:
            df_data = pd.DataFrame(data=df_array, columns=fcols)
        else:
            cols = []
            for i in range(len(df_array[0])): cols.append('col' + str(i))
            df_data = pd.DataFrame(data=df_array, columns=cols)
    
    targetCol = -1 # index of target column

    # extract column name before change into np.array
    dataCols = np.array(df_data.columns)

    # change df_data into np.array
    df_data = df_data.to_numpy()
    df_data = pd.DataFrame(df_data, columns=dataCols)
    print('\n<<< [0] df_data.shape >>>')
    print('columns : ' + str(df_data.columns))
    print('shape   : ' + str(df_data.shape))

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
            if col != targetCol:
                dataPart.append(col)
                extractCols.append(col)
                extractColInfo.append('data')
                dataPartAdded = True

        else: # test mode -> targetCol does not exist
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
    if isTrain == True: dataSet = {'data':df_data[dataPart], 'target':df_data[targetPart]}
    else: dataSet = {'data':df_data[dataPart]}
    
    dataSetDF = df_data[extractCols]

    # change dataSetDF into float type
    try:
        dataSetDF = dataSetDF.astype(float)
    except:
        doNothing = 0 # do nothing

    # print dataFrame
    print('\n<<< [2] dataSetDF >>>')
    print(dataSetDF)

     # again, change dataSetDF into float type
    try:
        dataSetDF = dataSetDF.astype(float)
    except:
        doNothing = 0 # do nothing

    # apply log for extractCols if useLog is true
    if useLog == True:

        # prevent error when frequentWords is None
        if frequentWords == None: frequentWords = []

        # actual column name is 'CT_' + each frequent word
        CTfreqWords = []
        for i in range(len(frequentWords)): CTfreqWords.append('CT_' + frequentWords[i])

        # using log: x -> log2(x + logConstant)
        for col in extractCols + CTfreqWords:

            if col == target or col[:3] == 'CT_': continue # except for target column and CT_ columns
            
            for i in range(len(dataSetDF)):
                dataSetDF.at[i, col] = math.log(max(0, dataSetDF.at[i, col]) + logConstant, 2)

        print('\n<<< [2-3] dataSetDF log applied >>>')
        print(dataSetDF)

    # again, change dataSetDF into float type
    try:
        dataSetDF = dataSetDF.astype(float)
    except:
        doNothing = 0 # do nothing

    print('\n<<< [2-4] dataSetDF original >>>')
    print(dataSetDF)

    # remove rows not included in 'rows'
    if rows != None:
        dataSetDF = dataSetDF.iloc[rows]

        print('\n<<< [2-5] dataSetDF after row extraction >>>')
        print(dataSetDF)

    print('')
    print('+========================+')
    print('|  Exit : makeDataFrame  |')
    print('+========================+')

    # return dataFrame
    return (dataSetDF, targetCol)
