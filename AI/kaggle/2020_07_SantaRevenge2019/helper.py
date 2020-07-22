import math
import datetime as dt

DATES = 100
FAMILIES = 6000
CHOICES = 10

# copy 2-d array
def arrayCopy(array):
    height = len(array)
    width = len(array[0])
    result = [[0] * width for j in range(height)]
    
    for i in range(height):
        for j in range(width):
            result[i][j] = array[i][j]

    return result

# data : raw file (by .readlines())
# start: start row of data
def makeTable(data, start):
    result = []
    for i in range(start, len(data)):
        
        thisRow = data[i].split('\n')[0].split(',')
        temp = []
        for j in range(len(thisRow)): temp.append(int(thisRow[j]))
        result.append(temp)

    return result

# get occupancy array
def getOccupancy(famData, subData):

    # people attending workshop on each day (occupancy)
    occupancyArray = []
    for i in range(DATES+1): occupancyArray.append(0)
    for i in range(FAMILIES): occupancyArray[subData[i][1]] += famData[i][11]
    return occupancyArray

# feasible
def isFeasible(famData, subData, occupancyArray):
    
    # check if feasible
    for i in range(1, DATES+1):
        if occupancyArray[i] < 125 or occupancyArray[i] > 300:
            # print(str(i) + '-th element is ' + str(occupancyArray[i]))
            return False
    return True

# calculate preference cost by using 2 tables, famData and subData
# suppose that it is feasible and there is an occupancy array
def prefCost(famData, subData, ocA):

    result = 0
    for i in range(FAMILIES):
        
        date = subData[i][1]
        members = famData[i][11]

        if famData[i][1] == date: # choice_0
            result += 0
        elif famData[i][2] == date: # choice_1
            result += 50
        elif famData[i][3] == date: # choice_2
            result += (50 + 9 * members)
        elif famData[i][4] == date: # choice_3
            result += (100 + 9 * members)
        elif famData[i][5] == date: # choice_4
            result += (200 + 9 * members)
        elif famData[i][6] == date: # choice_5
            result += (200 + 18 * members)
        elif famData[i][7] == date: # choice_6
            result += (300 + 18 * members)
        elif famData[i][8] == date: # choice_7
            result += (300 + 36 * members)
        elif famData[i][9] == date: # choice_8
            result += (400 + 36 * members)
        elif famData[i][10] == date: # choice_9
            result += (500 + 235 * members)
        else: # otherwise
            result += (500 + 434 * members)
    
    return result

# calculate accounting penalty by using 2 tables, famData and subData
# suppose that it is feasible and there is an occupancy array
def accountPenalty(famData, subData, ocA):

    result = 0.0
    for i in range(1, DATES+1):
        for j in range(1, 6):
            val0 = (ocA[i] - 125)/400
            val1 = pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[min(i+j, 100)]))/(j*j)
            result += val0 * val1

    return result

# get score
def getScore(famData, subData):

    # get occupancy array
    ocA = getOccupancy(famData, subData)

    # if not feasible
    if not isFeasible(famData, subData, ocA): return 2147483647
    
    return prefCost(famData, subData, ocA) + accountPenalty(famData, subData, ocA)

# get score (suppose that it is feasible and there is occupancy array)
def getScoreFeasible(famData, subData, ocA):
    return prefCost(famData, subData, ocA) + accountPenalty(famData, subData, ocA)

# print table row by row (at most the limit)
def printRowByRow(table, limit):
    for i in range(min(len(table), limit)): print(table[i])
    if limit < len(table): print('...')

# write sample submission file by using table subData
def writeSampleSub(subData, fn):
    toWrite = ''
    for i in range(len(subData)):
        toWrite += str(subData[i][0]) + ',' + str(subData[i][1]) + '\n'

    subFile = open(fn + '.csv', 'w')
    subFile.write(toWrite)
    subFile.close()

# main
if __name__ == '__main__':
    
    # read family data
    familyData = open('family_data.csv', 'r')
    data0 = familyData.readlines()
    familyData.close()

    familyDataTable = makeTable(data0, 1)

    # read sample submission data
    sampleSub = open('sample_submission.csv', 'r')
    data1 = sampleSub.readlines()
    sampleSub.close()

    sampleSubDataTable = makeTable(data1, 1)

    # print table
    printRowByRow(familyDataTable, 100)
    printRowByRow(sampleSubDataTable, 100)

    # calculate cost
    occupancyArray = getOccupancy(familyDataTable, sampleSubDataTable)
    pref = prefCost(familyDataTable, sampleSubDataTable, occupancyArray)
    account = accountPenalty(familyDataTable, sampleSubDataTable, occupancyArray)
    score = getScore(familyDataTable, sampleSubDataTable)

    # check if feasible
    print(isFeasible(familyDataTable, sampleSubDataTable, occupancyArray))

    print('preference cost   : ' + str(pref))
    print('accounting penalty: ' + str(account))
    print('score             : ' + str(score))

    # time test
    before = dt.datetime.now().timestamp()
    for x in range(1000): pref = prefCost(familyDataTable, sampleSubDataTable, occupancyArray)
    after = dt.datetime.now().timestamp()
    print('time for calculating 1000 prefs for whole dataset: ' + str(after - before))

    before = dt.datetime.now().timestamp()
    for x in range(1000): account = accountPenalty(familyDataTable, sampleSubDataTable, occupancyArray)
    after = dt.datetime.now().timestamp()
    print('time for calculating 1000 acPens for whole dataset: ' + str(after - before))
    
    # write sample submission file
    writeSampleSub(sampleSubDataTable, 'sampleExample')
