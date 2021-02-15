import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    threshold = 0.798
    option = 1

    # read file
    testResult = RD.loadArray('test_output_to_submit.txt')

    # write final result
    finalResult = []
    for i in range(len(testResult)):

        # option 0: binary value
        if option == 0:
            if float(testResult[i][0]) >= threshold:
                finalResult.append([1])
            else:
                finalResult.append([0])

        # option 1: at least 0, at most 1
        elif option == 1:
            if float(testResult[i][0]) > 1:
                finalResult.append([1])
            elif float(testResult[i][0]) < 0:
                finalResult.append([0])
            else:
                finalResult.append([float(testResult[i][0])])

    # write file
    RD.saveArray('to_submit.txt', finalResult)
            
