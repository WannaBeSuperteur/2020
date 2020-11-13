import sys
import numpy as np
sys.path.insert(0, '../../../../AI_BASE')

import readData as RD
import deepLearning_main as DL
import MAE_for_CRGoL as MAE

# read test output for n-sub mode
# idList         : list of test output IDs
# dataSize       : size of each test data (=400)
# testResult     : test result file name
# threshold      : live(=1) if >=threshold, dead(=0) if <threshold
# resultFileName : final result file name
def readTestOutput(idList, dataSize, testResult, threshold, resultFileName):

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
    readTestOutput('test_id_sub_0.txt', 400, 'test_output_n_sub_0.txt', 0.41, 'final_0.csv')
    readTestOutput('test_id_sub_1.txt', 400, 'test_output_n_sub_1.txt', 0.41, 'final_1.csv')
    readTestOutput('test_id_sub_2.txt', 400, 'test_output_n_sub_2.txt', 0.41, 'final_2.csv')
    readTestOutput('test_id_sub_3.txt', 400, 'test_output_n_sub_3.txt', 0.41, 'final_3.csv')
    readTestOutput('test_id_sub_4.txt', 400, 'test_output_n_sub_4.txt', 0.41, 'final_4.csv')
