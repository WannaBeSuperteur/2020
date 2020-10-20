import numpy as np
import math
import pandas as pd
import printData as PD_

# k Nearest Neighbor algorithm
# dfTrain    : dataframe for training data
# dfTest     : dataframe for test data
# targetCol  : target column for dfTrain
# targetIndex: index for the target column
# k          : number of neighbors
# useAverage : return average value
def kNN(dfTrain, dfTest, targetCol, targetIndex, k, useAverage):
    print('')
    print('+=======================+')
    print('|    Function : kNN     |')
    print('+=======================+')

    # count of each value for training data
    targetVals = list(set(dfTrain[targetCol].values)) # set of target values
    classCount = dfTrain[targetCol].value_counts() # class count for each target value

    print('\n<<< [17] dataFrame for training >>>')
    print(dfTrain)
    print('\n<<< [18] dataFrame for test >>>')
    print(dfTest)

    # convert to numpy array
    dfTrain = np.array(dfTrain)
    dfTest = np.array(dfTest)
    
    # kNN classification result
    result = []
    resultT = [] # transport of result

    # mark the result using k-nearest neighbor
    for i in range(len(dfTest)): # for each test data
        if i % 10 == 0: print('test data ' + str(i))
        
        thisTestData = dfTest[i]

        # [distance between the test data and each training data, mark]
        distAndMark = []

        # for each training data
        for j in range(len(dfTrain)):
            thisTrainData = dfTrain[j]

            # calculate distance from the test data
            thisDistSquare = 0
            for l in range(len(dfTrain[0])):
                if l == targetIndex: continue
                thisDistSquare = thisDistSquare + pow(thisTestData[l] - thisTrainData[l], 2)

            # add to distAndMark
            distAndMark.append([math.sqrt(thisDistSquare), thisTrainData[targetIndex]])

        # sort distAndMark array
        distAndMark = sorted(distAndMark, key=lambda x:x[0], reverse=False)

        # count the vote for each class (using weight = len(dfTrain)/trainCount)
        vote = {} # vote result for each class: [class targetVals[j], vote score of targetVals[j]]
        for j in range(len(classCount)): vote[targetVals[j]] = 0 # initialize dictionary vote
        for j in range(k): # count the vote using k nearest neighbors
            thisMark = distAndMark[j][1] # mark of this 'neighbor'
            vote[thisMark] = vote[thisMark] + len(dfTrain) / classCount[thisMark]

        # find max vote item
        if useAverage == True:
            sumOfVote = 0.0 # sum of (vote weight)
            sumOfKeyVal = 0.0 # sum of (key value)*(vote weight)
            
            for key in vote.keys():
                sumOfVote += vote[key]
                sumOfKeyVal += key * vote[key]

            avgVoteVal = sumOfKeyVal / sumOfVote

            # append the average vote result (=prediction) to result array
            result.append(avgVoteVal)
            resultT.append([avgVoteVal])

        # use average vote value
        else:
            largestVoteVal = -1 # number of votes of largest voted target value
            largestVoteTargetVal = -1 # largest voted target value

            # key: class targetVals[j], value: vote score of targetVals[j]
            for key in vote.keys():
                value = vote[key]
                
                if value > largestVoteVal:
                    largestVoteVal = value
                    largestVoteTargetVal = key

            # append the largest vote result (=prediction) to result array
            result.append(largestVoteTargetVal)
            resultT.append([largestVoteTargetVal])

    # add vote result value
    # https://rfriend.tistory.com/352
    dfTest = np.column_stack([dfTest, resultT])

    # display as chart
    title = '(kNN) test data prediction'

    # print data as 2d or 3d space
    if len(dfTest[0]) == 3: # 2 except for target col
        PD_.printDataAsSpace(2, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'target']), title)
    elif len(dfTest[0]) == 4: # 3 except for target col
        PD_.printDataAsSpace(3, pd.DataFrame(dfTest, columns=['pca0', 'pca1', 'pca2', 'target']), title)
            
    # return the result array
    return result
