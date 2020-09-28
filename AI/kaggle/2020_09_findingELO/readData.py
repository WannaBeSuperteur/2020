import numpy as np

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
        for j in range(cols):
            if j < cols-1: result += _2dArray[i][j] + '\t'
            else: result += _2dArray[i][j]

        result += '\n'

    # write to file
    f = open(fn, 'w')
    f.write(result)
    f.close()

# load result array
def loadArray(fn):
    
    # read *.pgn file
    f = open(fn, 'r')
    fLines = f.readlines()
    rows = len(fLines)
    f.close()

    result = []

    # append info to the result array
    for i in range(rows):
        thisRow = fLines[i].split('\n')[0].split('\t')
        cols = len(thisRow)

        # add this row
        temp = []
        for j in range(cols): temp.append(thisRow[j])
        result.append(temp)

    # return the result array
    return result

# test
if __name__ == '__main__':
    (trainPgn, testPgn) = readPGN('data.pgn', '[Event', [[0, 8, 2], [6, 9, 2], [7, 11, 2], [8, 11, 2]],
                                  [[0, 8, 2], [6, 9, 2]], 25000)
    
    saveArray('data_trainPgn.txt', trainPgn)
    saveArray('data_testPgn.txt', testPgn)
    
    print(np.array(loadArray('data_trainPgn.txt')))
    print(np.array(loadArray('data_testPgn.txt')))
