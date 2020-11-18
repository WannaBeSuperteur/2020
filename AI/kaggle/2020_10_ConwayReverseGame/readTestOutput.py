import sys
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
# dataSize       : size of each test data (=400)
# testResult     : test result file name
# resultFileName : final result file name
def readTestOutput(idList, dataSize, testResult, resultFileName):

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
    for i in range(len(resultList)): resultArray.append(float(resultList[i].split('\n')[0]))

    # make the result
    finalResult = ''
    for i in range(int(len(resultList)/dataSize)): # for each output board
        if i % 100 == 0: print('make result : ' + str(i))
        
        finalResult += str(idArray[i]) + ','
        for j in range(dataSize):
            finalResult += str(resultArray[i * dataSize + j])
            if j < dataSize-1: finalResult += ','
        finalResult += '\n'

    # write to the result file
    final = open(resultFileName, 'w')
    final.write(finalResult)
    final.close()

if __name__ == '__main__':
    
    try:
        sub0 = RD.loadArray('final_0.csv', ',')
        sub1 = RD.loadArray('final_1.csv', ',')
        sub2 = RD.loadArray('final_2.csv', ',')
        sub3 = RD.loadArray('final_3.csv', ',')
        sub4 = RD.loadArray('final_4.csv', ',')
    except:
        readTestOutput('test_id_sub_0.txt', 400, 'test_output_n_sub_0.txt', 'final_0.csv')
        readTestOutput('test_id_sub_1.txt', 400, 'test_output_n_sub_1.txt', 'final_1.csv')
        readTestOutput('test_id_sub_2.txt', 400, 'test_output_n_sub_2.txt', 'final_2.csv')
        readTestOutput('test_id_sub_3.txt', 400, 'test_output_n_sub_3.txt', 'final_3.csv')
        readTestOutput('test_id_sub_4.txt', 400, 'test_output_n_sub_4.txt', 'final_4.csv')

    # convert into 0 or 1 according to threshold
    threshold = 0.41
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
