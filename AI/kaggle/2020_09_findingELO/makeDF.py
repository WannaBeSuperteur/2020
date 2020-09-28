import pandas as pd
import numpy as np
import readData as RD

# fn           : file name
# ftype        : file type (json, csv, txt)
# fcols        : columns (if ftype is 'txt' and fcols is None, use default name)

# isTrain      : training(True) or not(False)
# target       : target column if isTrain is True
# tfCols       : columns that contains True or False values
# exceptCols   : using the columns except for them
# useLog       : using log for making dataFrame
# logConstant  : x -> log2(x + logConstant)
# specificCol  : column -> columns that indicate the number of appearance of frequent words
# frequentWords: list of frequent words
def makeDataFrame(fn, ftype, fcols, isTrain, target, tfCols, exceptCols, useLog, logConstant, specificCol, frequentWords):
    print('\n ######## makeDataFrame function ########')
    
    # open and show plt data
    if ftype == 'json': # type is 'json'
        jf = open(fn, 'r')
        df_loaded = json.load(jf)
        df_data = pd.DataFrame(df_loaded)
        
    elif ftype == 'csv': # type is 'csv'
        df_data = pd.read_csv(fn)
        
    elif ftype == 'txt': # type is 'txt'
        df_array = RD.loadArray(fn)
        if fcols != None: df_data = pd.DataFrame(data=df_array, columns=fcols)
        else:
            cols = []
            for i in range(len(df_array[0])): cols.append('col' + str(i))
            df_data = pd.DataFrame(data=df_array, columns=cols)
    
    targetCol = -1 # index of target column

    # True to 1, False to 0, and decimal to 0
    # https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false
    tfColsIndex = []
    for i in range(len(tfCols)): tfColsIndex.append(-1) # column indices of tfCols

    # initialize tfColsIndex
    for i in range(len(df_data.columns)):
        for j in range(len(tfCols)):
            if df_data.columns[i] == tfCols[j]: tfColsIndex[j] = i

    # extract column name before change into np.array
    dataCols = np.array(df_data.columns)

    # change df_data into np.array
    df_data = df_data.to_numpy()

    # modify values: True to 1, False to 0, and decimal to 0
    for x in range(len(tfCols)):

        # True to 1, False to 0, and decimal to 0
        for i in range(len(df_data)):
            if str(df_data[i][tfColsIndex[x]]) == 'True':
                df_data[i][tfColsIndex[x]] = 1
            else:
                df_data[i][tfColsIndex[x]] = 0

    df_data = pd.DataFrame(df_data, columns=dataCols)
    print('\n<<< [0] df_data.shape >>>\n' + str(df_data.shape))

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

    # add text column
    # column -> columns that indicate the number of appearance of frequent words
    if specificCol != None and frequentWords != None:

        # add specificCol to the dataFrame
        dataSetDF = df_data[extractCols + [specificCol]]

        # appearanceOfFrequentWordsTest(dataSetDF, specificCol)
        dataSetDF = appearanceOfFrequentWords(dataSetDF, specificCol, frequentWords)

        print('\n<<< [2-1] dataSetDF.columns with appearance of frequent words >>>')
        print(dataSetDF.columns)

        print('\n<<< [2-2] dataSetDF with appearance of frequent words >>>')
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

    # return dataFrame
    return (dataSetDF, targetCol)
