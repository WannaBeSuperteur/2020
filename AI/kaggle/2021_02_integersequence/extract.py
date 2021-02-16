import sys
sys.path.insert(0, '../../AI_BASE')

import math
import copy
import numpy as np
import readData as RD
import deepLearning_main as DL

if __name__ == '__main__':
    f = open('train.csv', 'r')
    fl = f.readlines()
    fl = fl[1:]
    f.close()

    # input size
    inputSize = 30

    # training data 0: original
    input_0 = []
    output_0 = []

    # training data 1: difference
    #                  input size = inputSize - 1
    input_1 = []
    output_1 = []

    # training data 2: log(x+1) if x >= 0 and -log((-x)+1) if x < 0
    input_2 = []
    output_2 = []

    # training data 3: log(difference+1) if x >= 0 and -log((-difference)+1) if x < 0
    #                  input size = inputSize - 1
    input_3 = []
    output_3 = []
    
    for i in range(min(len(fl), 40000)):
        if i % 250 == 0: print(i)
        
        sequence = fl[i].split('"')[1].split(',')

        while len(sequence) < inputSize + 1:
            sequence = [0] + sequence

        for j in range(len(sequence) - inputSize):
            inputData = sequence[j : j+inputSize]
            outputData = sequence[j+inputSize : j+inputSize+1]

            # convert to integer
            for k in range(len(inputData)):
                inputData[k] = int(inputData[k])
            outputData[0] = int(outputData[0])

            # compute difference
            input_dif = []
            output_dif = []
            
            for k in range(inputSize-1): input_dif.append(inputData[k+1] - inputData[k])
            output_dif.append(outputData[0] - inputData[inputSize-1])

            # compute log
            logInputData = []
            logOutputData = []
            
            for k in range(inputSize):
                if inputData[k] >= 0:
                    logInputData.append(math.log(inputData[k] + 1, 10))
                else:
                    logInputData.append(-math.log(-inputData[k] + 1, 10))

            if outputData[0] >= 0:
                logOutputData.append(math.log(outputData[0] + 1, 10))
            else:
                logOutputData.append(-math.log(-outputData[0] + 1, 10))

            # compute log_difference
            logDifInputData = []
            logDifOutputData = []

            for k in range(inputSize-1):
                if input_dif[k] >= 0:
                    logDifInputData.append(math.log(input_dif[k] + 1, 10))
                else:
                    logDifInputData.append(-math.log(-input_dif[k] + 1, 10))

            if output_dif[0] >= 0:
                logDifOutputData.append(math.log(output_dif[0] + 1, 10))
            else:
                logDifOutputData.append(-math.log(-output_dif[0] + 1, 10))
                    

            # input/output 0: original
            input_0.append(copy.deepcopy(inputData))
            output_0.append(copy.deepcopy(outputData))

            # input/output 1: difference
            input_1.append(copy.deepcopy(input_dif))
            output_1.append(copy.deepcopy(output_dif))

            # input/output 2: log
            input_2.append(copy.deepcopy(logInputData))
            output_2.append(copy.deepcopy(logOutputData))

            # input/output 3: log(difference)
            input_3.append(copy.deepcopy(logDifInputData))
            output_3.append(copy.deepcopy(logDifOutputData))

    # save as file
    RD.saveArray('input_0.txt', input_0, '\t', 500)
    RD.saveArray('output_0.txt', output_0, '\t', 500)

    RD.saveArray('input_1.txt', input_1, '\t', 500)
    RD.saveArray('output_1.txt', output_1, '\t', 500)

    RD.saveArray('input_2.txt', input_2, '\t', 500)
    RD.saveArray('output_2.txt', output_2, '\t', 500)

    RD.saveArray('input_3.txt', input_3, '\t', 500)
    RD.saveArray('output_3.txt', output_3, '\t', 500)
