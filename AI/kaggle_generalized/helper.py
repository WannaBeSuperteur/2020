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
