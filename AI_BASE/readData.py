import numpy as np
import math

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
def splitArray(fn, columns, exceptForFirstColumn):

    # load array
    originalArray = loadArray(fn)
    if exceptForFirstColumn == True: originalArray = originalArray[1:]

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
        saveArray(fn[:len(fn)-4] + '_sub_' + str(i) + '.txt', arrayByColValues[i][1])

    # return splitted array
    return colValues

# merge test result using IDs of original training data file (test only, not for validation)
# original : original test data file name (without '.txt')
# IDcol    : ID column, for example 0
# files    : number of files created by splitArray function
def mergeTestResult(original, IDcol, files):
    merged = []

    for i in range(files):

        # load test result array from result file
        result = loadArray('result_split_' + str(i) + '.csv', ',')
        result = result[1:] # remove first value with column title

        # read corresponding ID
        test = loadArray(original + '_sub_' + str(i) + '.txt')
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
    saveArray('result_split_final.txt', merged)

# lineStart : start with this symbol, for example (data.pgm), it is '[Event'
# nsTrain   : for example (data.pgm), to obtain 'Event', 'Result', 'WhiteElo' and 'BlackElo',
#             it is [[0, 8, 2], [6, 9, 2], [7, 11, 2], [8, 11, 2]]
# nsTest    : for example (data.pgm), to obtain 'Event', 'Result', 'WhiteElo' and 'BlackElo',
#             it is [[0, 8, 2], [6, 9, 2]]
# trainN    : number of training data, for example (data.pgm), it is 25000
def readPGN(fn, lineStart, nsTrain, nsTest, trainN):

    trainResult = [] # training data
    testResult = [] # test data

    # read *.pgn file
    f = open(fn, 'r')
    fLines = f.readlines()
    lines = len(fLines)
    f.close()

    # for each line
    for i in range(lines):

        # append 'row' in the dataset, to the result array
        if fLines[i][:len(lineStart)] == lineStart:
            
            temp = [] # data to save

            if len(trainResult) < trainN: # training data
                for j in range(len(nsTrain)):
                    index = nsTrain[j][0] # distance from the start line from the file
                    leftN = nsTrain[j][1] # except for left N chars
                    rightN = nsTrain[j][2] # except for right N chars

                    fLines[i+index] = fLines[i+index].split('\n')[0] # remove next-line character
                    value = fLines[i+index] # original value of this line from the file

                    temp.append(value[leftN:len(value)-rightN]) # append to the temp array
                    
            else: # test data
                for j in range(len(nsTest)):
                    index = nsTest[j][0] # distance from the start line from the file
                    leftN = nsTest[j][1] # except for left N chars
                    rightN = nsTest[j][2] # except for right N chars

                    fLines[i+index] = fLines[i+index].split('\n')[0] # remove next-line character
                    value = fLines[i+index] # original value of this line from the file

                    temp.append(value[leftN:len(value)-rightN]) # append to the temp array

            # append to the result array (training/test)
            if len(trainResult) < trainN: trainResult.append(temp) # training data
            else: testResult.append(temp) # test data

    # return the result array
    return (trainResult, testResult)

# save result array
def saveArray(fn, _2dArray):
    
    result = ''
    rows = len(_2dArray) # rows of 2d array
    cols = len(_2dArray[0]) # cols of 2d array

    # append to result
    for i in range(rows):
        if i % 100 == 0: print('row ' + str(i))
        for j in range(cols):
            if j < cols-1: result += str(_2dArray[i][j]) + '\t'
            else: result += str(_2dArray[i][j])

        result += '\n'

    # write to file
    f = open(fn, 'w')
    f.write(result)
    f.close()

# load result array with splitter
def loadArray(fn, splitter='\t'):
    
    # read *.pgn file
    f = open(fn, 'r')
    fLines = f.readlines()
    rows = len(fLines)
    f.close()

    result = []

    # append info to the result array
    for i in range(rows):
        thisRow = fLines[i].split('\n')[0].split(splitter)
        cols = len(thisRow)

        # add this row
        temp = []
        for j in range(cols): temp.append(thisRow[j])
        result.append(temp)

    # return the result array
    return result

# get loged value
def getLogVal(val):
    try: x = float(val)
    except: x = 0
                
    if x < 0: return -math.log(1-x, 10)
    else: return math.log(1+x, 10)

# convert text column into one-hot column (only used for 2d array)
def textColumnToOneHot(array, colIndex, colItems):
    resultArray = [] # final array to return

    rows = len(array) # number of rows in original array
    columns = len(array[0]) # number of columns in original array

    # make resultArray
    for i in range(rows):
        thisRow = [] # each row of resultArray

        # for each atomic cell in original array
        for j in range(columns):

            # append as original for non-colIndex column
            if j != colIndex: thisRow.append(array[i][j])

            # append as one-hot for colIndex column
            else:
                for k in range(itemKinds):
                    if colItems[k] == array[i][j]: thisRow.append(1)
                    else: thisRow.append(0)

        # append to finalResult
        resultArray.append(thisRow)

    # return final result array
    return resultArray

# test
if __name__ == '__main__':
    
    # read from original data
    (trainPgn, testPgn) = readPGN('data.pgn', '[Event', [[0, 8, 2], [6, 9, 2], [7, 11, 2], [8, 11, 2]],
                                  [[0, 8, 2], [6, 9, 2]], 25000)

    # read from stockfish.csv
    fs = open('stockfish.csv', 'r')
    fsLines = fs.readlines()
    rows = len(fsLines)
    fs.close()

    for i in range(50000):
        fsLines[i+1] = fsLines[i+1].split('\n')[0] # remove new-line character from the line
        thisLineSplit = fsLines[i+1].split(',')[1].split(' ') # split moveScores
        gameLength = len(thisLineSplit)

        # extract data from moveScores
        gameData = []
        N = 8 # number of points to extract data
        for j in range(N):
            try:
                gameData.append(thisLineSplit[int(j)])
            except:
                gameData.append(0)

        # handle NA data
        for j in range(len(gameData)):
            if gameData[j] == 'NA': gameData[j] = gameData[j-1]

        # add this data to trainPgn or testPgn
        if i < 25000: # training data -> add to trainPgn
            for j in range(len(gameData)):
                trainPgn[i].append(getLogVal(gameData[j]))
                
        else: # test data -> add to testPgn
            for j in range(len(gameData)):
                testPgn[i-25000].append(getLogVal(gameData[j]))

    # print the result (before conversion)
    print('\n <<< trainPgn (before conversion) >>>')
    print(np.array(trainPgn))

    print('\n <<< testPgn (before conversion) >>>')
    print(np.array(testPgn))

    # list of item for specified column (colIndex = 1)
    colIndex = 1
    
    thisCol = np.array(trainPgn)[:,colIndex]
    colItems = list(set(thisCol))
    itemKinds = len(colItems)
    
    print('\n <<< item list >>>')
    print(colItems)

    # convert index 1 column (win-lose) to one-hot
    trainPgn = textColumnToOneHot(trainPgn, colIndex, colItems)
    testPgn = textColumnToOneHot(testPgn, colIndex, colItems)

    # print the result (after conversion)
    print('\n <<< trainPgn (after conversion) >>>')
    print(np.array(trainPgn))

    print('\n <<< testPgn (after conversion) >>>')
    print(np.array(testPgn))

    # save array
    saveArray('data_trainPgn.txt', trainPgn)
    saveArray('data_testPgn.txt', testPgn)

    print('\n <<< trainPgn File >>>')
    print(np.array(loadArray('data_trainPgn.txt')))

    print('\n <<< testPgn File >>>')
    print(np.array(loadArray('data_testPgn.txt')))
