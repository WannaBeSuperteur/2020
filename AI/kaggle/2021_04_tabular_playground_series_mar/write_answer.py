import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    # read file
    testResult = RD.loadArray('test_output.txt')

    # write final result
    finalResult = []
    
    for i in range(len(testResult)):
        finalResult.append([float(testResult[i][0])])

    # write file
    RD.saveArray('to_submit.txt', finalResult)
            
