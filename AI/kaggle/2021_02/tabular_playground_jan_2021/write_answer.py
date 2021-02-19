import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    threshold = 0.798
    option = 1

    # read file
    testResult = RD.loadArray('test_output.txt')

    # write final result
    finalResult = []
    
    for i in range(len(testResult)):
        finalResult.append([float(testResult) + 8])

    # write file
    RD.saveArray('to_submit.txt', finalResult)
            
