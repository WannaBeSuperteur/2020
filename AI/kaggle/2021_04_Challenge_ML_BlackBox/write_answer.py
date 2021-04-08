import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import copy

# write final result
def getFinalResult(testResult):
    finalResult = []
    
    for i in range(len(testResult)):
        finalResult.append(np.argmax(testResult[i]) + 1)

    return finalResult

if __name__ == '__main__':

    files = 4

    # [result0, result1, ...]
    # where each element resultX is [1, 3, 4, 2, 9, 6, 5, 7, ...] for example
    finalResults = []

    # sum of test results
    sumTestResults = []

    # read file
    for i in range(4):
        testResult = np.array(RD.loadArray('test_output_' + str(i) + '.txt'))[:, :9].astype(float)

        if i == 0:
            sumTestResults = np.array(copy.deepcopy(list(testResult)))
        else:
            sumTestResults = sumTestResults + np.array(copy.deepcopy(list(testResult)))
        
        finalResult = getFinalResult(testResult)
        finalResults.append(finalResult)

    # save the sum of test results
    RD.saveArray('sumTestResults.txt', sumTestResults)

    # write final result
    # USE THE RIGHTMOST COLUMN OF to_submit.txt AS FINAL RESULT
    finalResults.append(getFinalResult(sumTestResults))
    RD.saveArray('to_submit.txt', np.array(finalResults).T)
            
