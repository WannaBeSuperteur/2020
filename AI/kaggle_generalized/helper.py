import math
import numpy as np
import time
from datetime import datetime

def isNumber(s):
    try:
        float(s)
        return True
    except:
        return False

# print rounded array
def roundedArray(array, n):
    roundedArray = []
    for i in range(len(array)):
        try:
            roundedArray.append(round(array[i], n))
        except:
            roundedArray.append(array[i])
    return roundedArray

# return timestamp as date
def timestamp_type0(s):
    ts = time.mktime(datetime.strptime(s, '%Y-%m-%d').timetuple())
    return int(ts / 86400)
def timestamp_type1(s):
    ts = time.mktime(datetime.strptime(s, '%m/%d/%Y').timetuple())
    return int(ts / 86400)

# read training/test data from *.csv file
# cols_           : list of columns designated as training/test data, e.g. [0, 1, 3, 5, 6, 10]
# type_           : type of each column of training/test data, e.g. [0, 1, 2, 0, 0, 1]
# makeNullList    : make null list for removing rows with NULL numeric/date values in train/test data
# existingNullList: existing null list (list of rows with NULL numeric/date values)
# nullValueArray  : array of alternative value when the test input value is null, e.g. [0, 0, 0, 0, 0, 0]
def getDataFromFile(fn, splitter, cols_, type_, makeNullList, existingNullList, nullValueArray):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    if makeNullList == True: nullList = [] # list of rows with NULL numeric/date values 

    result = []
    for i in range(1, len(flines)): # len(flines)

        # remove (do not consider -> continue) this row if it is in nullList
        if existingNullList != None:
            if i in existingNullList: continue

        # replace all ','s within "", to '#'
        if '"' in flines[i]:
            quoteCount = 0
            for j in range(len(flines[i])):
                if flines[i][j] == '"': quoteCount += 1
                elif flines[i][j] == ',' and quoteCount % 2 == 1: flines[i] = flines[i][:j] + '#' + flines[i][j+1:]

        # parse each row
        row = flines[i].split('\n')[0].split(splitter)
        isNull = False # is this row NULL?

        # for each column designated as training/test data
        for j in range(len(cols_)):

            thisColIndex = cols_[j] # index of this column in the row

            # check if this value is NULL when it is not a text value
            if type_[j] != 2:
                if row[thisColIndex] == 'NULL' or row[thisColIndex] == 'null' or row[thisColIndex] == 'Null':

                    isNull = True
                    if makeNullList == True: nullList.append(i) # append this row to nullList
                    elif existingNullList == None:
                        row[thisColIndex] = nullValueArray[j] # use nullValue for test data
                        isNull = False # do not consider it as null value
                    
                    continue

            # if this value is numeric
            if type_[j] == 0 or type_[j] == 3 or type_[j] == 4 or type_[j] == 5 or type_[j] == 6 or type_[j] == 7:

                if type_[j] == 3 or type_[j] == 5: # using log
                    row[thisColIndex] = math.log(float(row[thisColIndex]), 2) # using log-ed value
                elif type_[j] == 6 or type_[j] == 7: # using log(x+1)
                    row[thisColIndex] = math.log(float(row[thisColIndex])+1.0, 2) # using log(x+1)-ed value
                else: # not using log
                    row[thisColIndex] = float(row[thisColIndex]) # using original value

            # if this value is a date
            elif type_[j] == 1:
                row[thisColIndex] = timestamp_type0(row[thisColIndex]) # return the timestamp of data (yyyy-mm-dd)
            elif type_[j] == 8:
                row[thisColIndex] = timestamp_type1(row[thisColIndex]) # return the timestamp of data (mm/dd/yyyy)

        # append to result if the row is not NULL
        if isNull == False: result.append(row)

    if makeNullList == True: return (result, nullList)
    else: return result

# sub-function for categorizeData function
# return true when data satisfies condition, otherwise return false
def dataCondition(data, condition):

    # '>X', '>=X', '<X', '<=X' and '<>X'
    if condition[:2] == '>=' and len(condition) >= 3:
        if float(data) >= float(condition[2:]): return True
    elif condition[:2] == '<=' and len(condition) >= 3:
        if float(data) <= float(condition[2:]): return True
    elif condition[:2] == '<>' and len(condition) >= 3:
        if data != condition[2:]: return True
    elif condition[0] == '>' and len(condition) >= 2:
        if float(data) > float(condition[1:]): return True
    elif condition[0] == '<' and len(condition) >= 2:
        if float(data) < float(condition[1:]): return True

    # '*X*', '*X' and 'X*'
    elif condition[0] == '*' and condition[len(condition)-1] == '*' and len(condition) >= 3: # '*X*' : include X
        if condition[1:len(condition)-1] in data: return True
    elif condition[0] == '*' and len(condition) >= 2: # '*X' : ends with X
        if data[len(data)-len(condition)+1:] == condition[1:]: return True
    elif condition[len(condition)-1] == '*' and len(condition) >= 2: # 'X*' : starts with X
        if data[:len(condition)-1] == condition[:len(condition)-1]: return True

    # '!*X*', '!*X' and '!X*'
    elif condition[:2] == '!*' and condition[len(condition)-1] == '*' and len(condition) >= 4: # '!*X*' : does not include X
        if not condition[2:len(condition)-1] in data: return True
    elif condition[:2] == '!*' and len(condition) >= 3: # '!*X' : does not end with X
        if not data[len(data)-len(condition)+2:] == condition[2:]: return True
    elif condition[0] == '!' and condition[len(condition)-1] == '*' and len(condition) >= 3: # '!X*' : does not start with X
        if not data[:len(condition)-2] == condition[1:len(condition)-1]: return True

    # X (equal to)
    else:
        if data == condition: return True

    return False

# read training/test data from *.csv file
# cols_           : list of columns designated as training/test data, e.g. [0, 1, 3, 5, 6, 10]
# categorizeRules : rules for categorization for each column, e.g. [['>25', 'x', '>15', 'y', 'z'], ['A', 'x', 'B', 'y', 'z'], ...]
#                   '>X', '>=X', '<X', '<=X', 'X', '<>X': greater than/at least/less than/at most/equal to/not equal to X
#                   '*X*', '*X', 'X*': contains/end with/start with X
#                   '!*X*', '!*X', '!X*': do not contains/end with/start with X
def categorizeData(fn, splitter, cols_, categorizeRules):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    result = []
    for i in range(1, len(flines)): # len(flines)

        # replace all ','s within "", to '#'
        if '"' in flines[i]:
            quoteCount = 0
            for j in range(len(flines[i])):
                if flines[i][j] == '"': quoteCount += 1
                elif flines[i][j] == ',' and quoteCount % 2 == 1: flines[i] = flines[i][:j] + '#' + flines[i][j+1:]

        # parse each row
        row = []
        originalRow = flines[i].split('\n')[0].split(splitter) # each column of original data
        for j in range(len(cols_)): row.append(originalRow[cols_[j]])

        # for each column designated as training/test data
        for j in range(len(cols_)):

            # do not modify this row (remain original) if categorize rule does not exist
            if categorizeRules[j][0] == 'None_': continue

            # use the categorize rules to categorize the value of columns
            for k in range(int((len(categorizeRules[j])+1)/2)):

                # if row[thisColIndex] satisfies a condition
                if dataCondition(row[j], categorizeRules[j][2*k]):
                    row[j] = categorizeRules[j][2*k+1]
                    break

                # otherwise (not satisfy any condition)
                if k == int((len(categorizeRules[j])-1)/2): row[j] = categorizeRules[j][2*k]

        # append to result
        result.append(row)

    return result

# read data (NOT including cols not designated as input/output)
# from file using information in input_output_info.txt, in the form of
# *inputFileName
# trainCol0 categorizationRule0
# trainCol1 categorizationRule1
# ...
# trainColN categorizationRuleN
# *outputFileName
# testCol0 categorizationRule0
# testCol1 categorizationRule1
# ...
# testColN categorizationRuleN
# *testFileName
# testCol0 categorizationRule0
# testCol1 categorizationRule1
# ...
# testColN categorizationRuleN
# *testOutputFileName
def readFromFileCatg(fn):
    f = open(fn, 'r')
    ioInfo = f.readlines()
    f.close()

    mode = '' # mode (trainInput, trainOutput or testInput)
    inputFileName = '' # input train data file
    outputFileName = '' # output train data file
    testFileName = '' # test input data file
    testOutputFileName = '' # test output data file

    inputs = None # train input data
    outputs = None # train output data
    tests = None # test input data

    cols_ = [] # columns designated as input/output column
    categorizationRules = [] # categorization rules for these columns

    for i in range(len(ioInfo)):

        # change mode when the line starts with '*'
        if ioInfo[i][0] == '*':
            if mode == '': # train input file
                mode = 'trainInput'
                inputFileName = ioInfo[i].split('\n')[0][1:]
                
            elif mode == 'trainInput': # train output file
                mode = 'trainOutput'
                outputFileName = ioInfo[i].split('\n')[0][1:]
                inputs = categorizeData(inputFileName, ',', cols_, categorizationRules)
                cols_ = []
                categorizationRules = []
                
            elif mode == 'trainOutput': # test input file
                mode = 'testInput'
                testFileName = ioInfo[i].split('\n')[0][1:]
                outputs = categorizeData(outputFileName, ',', cols_, categorizationRules)
                cols_ = []
                categorizationRules = []
                
            elif mode == 'testInput': # test output file
                testOutputFileName = ioInfo[i].split('\n')[0][1:]
                tests = categorizeData(testFileName, ',', cols_, categorizationRules)

        # append the information to cols_ and categorizationRules
        else:
            ioInfoSplit = ioInfo[i].split('\n')[0].split(' ')
            
            cols_.append(int(ioInfoSplit[0]))

            temp = []
            for j in range(1, len(ioInfoSplit)): temp.append(ioInfoSplit[j])
            categorizationRules.append(temp)

    # return input, output and test data
    return (inputs, outputs, tests, testOutputFileName)

# return set of members of column colNum in the array
# example: array=[['a', 1], ['b', 1], ['c', 2], ['c', 3], ['a', 0]], colNum=0 -> ['a', 'b', 'c']
def makeSet(array, colNum):
    arraySet = []
    for i in range(len(array)): arraySet.append(array[i][colNum])
    arraySet = set(arraySet)
    arraySet = list(arraySet)

    return arraySet

# return set of members of column colNum in the array, using 2 arrays
def makeSet_(array0, array1, colNum0, colNum1):
    arraySet = []
    for i in range(len(array0)): arraySet.append(array0[i][colNum0])
    for i in range(len(array1)): arraySet.append(array1[i][colNum1])
    arraySet = set(arraySet)
    arraySet = list(arraySet)

    return arraySet

# sign of value
def sign(val):
    if val > 0: return 1
    elif val == 0: return 0
    else: return -1
