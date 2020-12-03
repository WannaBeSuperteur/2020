import sys
import math
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD
import deepLearning_main as DL
import MAE_for_CRGoL as MAE

# input  : separated final result (final_X.csv)
# output : combined final result (final.csv)

# files read by this code
# test_id_sub_X.txt       (open in readTestOutput)
# test_output_n_sub_X.txt (open in readTestOutput)
# final_X.csv             (loadArray in main)

# files written by this code
# final_X.csv (open in readTestOutput)
# final.csv   (saveArray in main)

# read test output for n-sub mode
# idList         : list of test output IDs
# dataSize       : size of each test data ( = 20*20 = 400)
# testResult     : test result file name
# resultFileName : final result file name
# weight         : weight matrix, same as MAE_for_CRGoL.py (ex: [1, 1, 1, 1, 2, 1, 1, 1, 1])
# n is derived from the size of array 'weight'
def readTestOutput(idList, dataSize, testResult, resultFileName, weight):

    size = int(math.sqrt(dataSize)) # number of rows/columns in original board
    n = len(weight) # number of values in array 'weight'
    rows = int(math.sqrt(n)) # number of rows in test output

    # read ID list
    f_id = open(idList, 'r')
    f_list = f_id.readlines()
    f_id.close()

    idArray = [] # list of ID (array)
    for i in range(len(f_list)): idArray.append(int(f_list[i].split('\n')[0]))

    # read test result
    resultFile = open(testResult, 'r')
    resultList = resultFile.readlines()
    resultFile.close()

    resultArray = [] # list of results (array)
    for i in range(len(resultList)):
        if i % 10000 == 0: print(str(i) + ' / ' + str(len(resultList)))

        # index of the line (indexed by i) for the board
        board_i = i % (size * size)
        board_i_y = int(board_i / size) # y-axis value of index i of the board
        board_i_x = board_i % size # x-axis value of index i of the board

        # find weighted prediction
        (resultValue, refList, refVals, refWeights) = MAE.getWeightedPrediction(n, i, size, board_i_y, board_i_x, weight, False, resultList)
        resultArray.append(resultValue)

    # make the result
    final = open(resultFileName, 'a')
    finalResult = ''
    
    for i in range(int(len(resultList)/dataSize)): # for each output board
        if i % 100 == 0: print('make result : ' + str(i))
        
        finalResult += str(idArray[i]) + ','
        for j in range(dataSize):
            finalResult += str(resultArray[i * dataSize + j])
            if j < dataSize-1: finalResult += ','
        finalResult += '\n'

        # write to the result file
        final.write(finalResult)
        finalResult = ''

    # close file
    final.close()

if __name__ == '__main__':

    use_n_sub = True # use n_sub mode
    size = 20 # the number of rows/columns in each input data
    inputLength = size * size # length of input vector

    # read array from final_X.csv

    # when using n_sub mode
    if use_n_sub == True:
        
        try:
            sub0 = RD.loadArray('final_0.csv', ',')
            sub1 = RD.loadArray('final_1.csv', ',')
            sub2 = RD.loadArray('final_2.csv', ',')
            sub3 = RD.loadArray('final_3.csv', ',')
            sub4 = RD.loadArray('final_4.csv', ',')
        except:
            weight = [1]
            readTestOutput('test_id_sub_0.txt', inputLength, 'test_output_n_sub_0.txt', 'final_0.csv', weight)
            readTestOutput('test_id_sub_1.txt', inputLength, 'test_output_n_sub_1.txt', 'final_1.csv', weight)
            readTestOutput('test_id_sub_2.txt', inputLength, 'test_output_n_sub_2.txt', 'final_2.csv', weight)
            readTestOutput('test_id_sub_3.txt', inputLength, 'test_output_n_sub_3.txt', 'final_3.csv', weight)
            readTestOutput('test_id_sub_4.txt', inputLength, 'test_output_n_sub_4.txt', 'final_4.csv', weight)
            
            sub0 = RD.loadArray('final_0.csv', ',')
            sub1 = RD.loadArray('final_1.csv', ',')
            sub2 = RD.loadArray('final_2.csv', ',')
            sub3 = RD.loadArray('final_3.csv', ',')
            sub4 = RD.loadArray('final_4.csv', ',')

    # when not using n_sub mode
    else:
        id0 = RD.loadArray('test_id_sub_0.txt')
        id1 = RD.loadArray('test_id_sub_1.txt')
        id2 = RD.loadArray('test_id_sub_2.txt')
        id3 = RD.loadArray('test_id_sub_3.txt')
        id4 = RD.loadArray('test_id_sub_4.txt')
        sub0 = RD.loadArray('test_output_sub_0.txt')
        sub1 = RD.loadArray('test_output_sub_1.txt')
        sub2 = RD.loadArray('test_output_sub_2.txt')
        sub3 = RD.loadArray('test_output_sub_3.txt')
        sub4 = RD.loadArray('test_output_sub_4.txt')

        # assertion: length of idX = length of subX
        assert(len(id0) == len(sub0))
        assert(len(id1) == len(sub1))
        assert(len(id2) == len(sub2))
        assert(len(id3) == len(sub3))
        assert(len(id4) == len(sub4))

        # merge: [test ouput data] -> [id, test output data]
        for i in range(len(sub0)): sub0[i] = id0[i] + sub0[i]
        for i in range(len(sub1)): sub1[i] = id1[i] + sub1[i]
        for i in range(len(sub2)): sub2[i] = id2[i] + sub2[i]
        for i in range(len(sub3)): sub3[i] = id3[i] + sub3[i]
        for i in range(len(sub4)): sub4[i] = id4[i] + sub4[i]

        # remove last null values
        for i in range(len(sub0)):
            if sub0[i][len(sub0[i])-1] == '': sub0[i].pop(len(sub0[i])-1)
        for i in range(len(sub1)):
            if sub1[i][len(sub1[i])-1] == '': sub1[i].pop(len(sub1[i])-1)
        for i in range(len(sub2)):
            if sub2[i][len(sub2[i])-1] == '': sub2[i].pop(len(sub2[i])-1)
        for i in range(len(sub3)):
            if sub3[i][len(sub3[i])-1] == '': sub3[i].pop(len(sub3[i])-1)
        for i in range(len(sub4)):
            if sub4[i][len(sub4[i])-1] == '': sub4[i].pop(len(sub4[i])-1)

    # convert into 0 or 1 according to threshold
    threshold = 0.38
    for i in [sub0, sub1, sub2, sub3, sub4]:
        for j in range(len(i)):
            i[j][0] = int(i[j][0])
            
            for k in range(1, len(i[j])):
                value = i[j][k]
                if float(i[j][k]) >= threshold: i[j][k] = 1 # above threshold -> live
                else: i[j][k] = 0 # below threshold -> dead

    # write final array
    finalArray = []
    
    print('for sub0')
    for i in range(len(sub0)): finalArray.append(sub0[i])
    print('for sub1')
    for i in range(len(sub1)): finalArray.append(sub1[i])
    print('for sub2')
    for i in range(len(sub2)): finalArray.append(sub2[i])
    print('for sub3')
    for i in range(len(sub3)): finalArray.append(sub3[i])
    print('for sub4')
    for i in range(len(sub4)): finalArray.append(sub4[i])
    print('finished')
    
    finalArray = sorted(finalArray, key=lambda x:x[0])
    RD.saveArray('final.csv', finalArray, ',')
