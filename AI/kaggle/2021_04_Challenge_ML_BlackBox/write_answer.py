import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

if __name__ == '__main__':

    # read file
    testResult = np.array(RD.loadArray('test_output.txt'))[:, :9]
    print('length =', len(testResult))

    # write final result
    finalResult = []
    
    for i in range(len(testResult)):
        finalResult.append([np.argmax(testResult[i]) + 1])

    # write file
    RD.saveArray('to_submit.txt', finalResult)
            
