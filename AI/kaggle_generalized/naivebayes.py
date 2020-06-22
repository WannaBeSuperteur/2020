import random
import numpy as np
import time
import math
from datetime import datetime
import helper

if __name__ == '__main__':
    printed = int(input('printed? (0 -> do not print)'))

    # [0] get data (onlt designated columns)
    # from file using information in input_output_info_nb.txt file, in the form of

    # *inputFileName
    # trainCol0 categorizationRule0
    # trainCol1 categorizationRule1
    # ...
    # trainColN categorizationRuleN
    # *outputFileName
    # testCol0 categorizationRule0
    # testCol1 categorizationRule1
    # ...
    # testColN categorizationRuleN
    # *testFileName
    # testCol0 categorizationRule0
    # testCol1 categorizationRule1
    # ...
    # testColN categorizationRuleN
    # *testOutputFileName
    (inputs, outputs, tests, testOutputFileName) = helper.readFromFileCatg('input_output_info_nb.txt')
    
    input_cols = len(inputs[0]) # number of columns of inputs
    output_cols = len(outputs[0]) # number of columns of outputs
    test_cols = len(tests[0]) # number of columns of tests

    # print if 'printed' is nonzero
    if printed != 0:
        print('')
        print(' ---- number of rows ----')
        print('input  size: ' + str(len(inputs)))
        print('output size: ' + str(len(outputs)))
        print('test   size: ' + str(len(tests)))
        print('')

        # print input, output, and test data
        print('\n ---- original input data ----\n')
        for i in range(len(inputs)): print(helper.roundedArray(inputs[i], 6))

        print('\n ---- original output data ----\n')
        for i in range(len(outputs)): print(helper.roundedArray(outputs[i], 6))

        print('\n ---- original test data ----\n')
        for i in range(len(tests)): print(helper.roundedArray(tests[i], 6))

    # [1] set of input, output and test data
    # set of value of each column of input+test data
    inputSet = []

    assert(input_cols == test_cols) # input data and test data has the same number of columns
    for i in range(input_cols): inputSet.append([]) # initialize array inputSet
    for i in range(input_cols):
        for j in range(len(inputs)): inputSet[i].append(inputs[j][i])
        for j in range(len(tests)): inputSet[i].append(tests[j][i])
        inputSet[i] = list(set(inputSet[i]))

    # set of value of each column of output data
    outputSet = []
    for i in range(output_cols): outputSet.append([]) # initialize array outputSet
    for i in range(output_cols):
        for j in range(len(outputs)): outputSet[i].append(outputs[j][i])
        outputSet[i] = list(set(outputSet[i]))

    # [2~3] learning

    # [2] total number and probability of Y = TRUE (or 1)
    total_Y = [] # total_Y[j] = total number of rows with Y = TRUE (or 1) for output column j = |Yj=TRUE|
    prob_Y = [] # prob_Y[j] = probability for Y = TRUE (or 1) for output column j = |Yj=TRUE|/(output rows)
    for i in range(output_cols):
        total_Y.append(0.0)
        prob_Y.append(0.0)

    # calculate total_Y and prob_Y
    for k in range(output_cols):
        for j in range(len(outputs)):
            if outputs[j][k] == 1 or outputs[j][k] == True or outputs[j][k] == 'True' or outputs[j][k] == 'TRUE':
                total_Y[k] += 1.0 # add 1.0 if TRUE
        prob_Y[k] = total_Y[k] / len(outputs) # calculate prob_Y[k] = total_Y[k]/(output rows) = P(Yk)

    # [3] calculate each probability

    # [3-0] initialize array
    # number of Xs for each input column, where Y = TRUE (1) for each output column
    # total_X_Y[i][k][j] = number of rows with (Xi=item_i(j) and Yk=TRUE) = |Xi=item_i(j),Yk=TRUE|
    total_X_Y = [[None]*output_cols for i in range(input_cols)]
    # number of Xs for each input column, where Y = FALSE (0) for each output column
    # total_X_notY[i][k][j] = number of rows with (Xi=item_i(j) and Yk=FALSE) = |Xi=item_i(j),Yk=FALSE|
    total_X_notY = [[None]*output_cols for i in range(input_cols)]
    
    # P(Xi=item_i(j)|Yk) for each input/output column
    # prob_X_Y[i][k][j] = P(Xi=item_i(j)|Yk) = |Xi=item_i(j),Yk=TRUE|/|Yk=TRUE| = total_X_Y[i][k][j] / total_Y[j]
    prob_X_Y = [[None]*output_cols for i in range(input_cols)]
    # P(Xi=item_i(j)|~Yk) for each input/output column
    # prob_X_notY[i][k][j] = P(Xi=item_i(j)|~Yk) = |Xi=item_i(j),Yk=FALSE|/|Yk=FALSE| = total_X_notY[i][k][j]/((output rows) - total_Y[j])
    prob_X_notY = [[None]*output_cols for i in range(input_cols)]

    # [3-1] initialize total_X_Y[i][k]s, total_X_notY[i][k]s, prob_X_Y[i][k]s and prob_X_notY[i][k]s
    for i in range(input_cols):
        for k in range(output_cols):
            
            total_X_Y[i][k] = []
            total_X_notY[i][k] = []
            prob_X_Y[i][k] = []
            prob_X_notY[i][k] = []

            # initialize as 0
            for j in range(inputSet):
                total_X_Y[i][k].append(0.0)
                total_X_notY[i][k].append(0.0)
                prob_X_Y[i][k].append(0.0)
                prob_X_notY[i][k].append(0.0)
        
    # [3-2] calculate total_X_Y's and total_X_notY's
    # total_X_Y[i][k][j] = number of rows with (Xi=item_i(j) and Yk=TRUE) = |Xi=item_i(j),Yk=TRUE|
    # total_X_notY[i][k][j] = number of rows with (Xi=item_i(j) and Yk=FALSE) = |Xi=item_i(j),Yk=FALSE|
    for row in range(len(outputs)): # for each row
        for i in range(input_cols): # for each input column
            for j in range(len(inputSet[i])): # for each input value
                if inputs[row][i] == inputSet[i][j]: # if Xi=item_i(j) then
                    for k in range(output_cols): # for each output column

                        # if Yk=TRUE
                        if outputs[row][k] == 1 or outputs[row][k] == True or outputs[row][k] == 'True' or outputs[row][k] == 'TRUE':
                            total_X_Y[i][k][j] += 1
                        # if Yk=False
                        else:
                            total_X_notY[i][k][j] += 1

    # [3-3] calculate P(Xi=item_i(j)|Yk)s and P(Xi=item_i(j)|~Yk)s
    # P(Xi=item_i(j)|Yk) = prob_X_Y[i][k][j] = |Xi=item_i(j),Yk=TRUE|/|Yk=TRUE| = total_X_Y[i][k][j] / total_Y[j]
    # P(Xi=item_i(j)|~Yk) = prob_X_notY[i][k][j] = |Xi=item_i(j),Yk=FALSE|/|Yk=FALSE| = total_X_notY[i][k][j]/((output rows) - total_Y[j])
    for i in range(input_cols):
        for k in range(output_cols):
            for j in range(len(prob_X_Y[i][k])): 
                prob_X_Y[i][k][j] = total_X_Y[i][k][j] / total_Y[j] # calculate P(Xi|Yj)s
                prob_X_notY[i][k][j] = total_X_notY[i][k][j] / (len(outputs)-total_Y[j]) # calculate P(Xi|~Yj)s
    
    # [4] test: calculate the probability
    # [Yk]  = P(Yk)     * P(X0=x0|Yk)       * P(X1=x1|Yk)       * ... * P(XN=xN|Yk)
    #       = P(Yk)     * P(X0=item0(j)|Yk) * P(X1=item1(j)|Yk) * ... * P(XN=itemN(j)|Yk) if x0=item0(j), x1=item1(j), ..., xN=itemN(j)
    #       = prob_Y[k] * prob_X_Y[0][k][j] * prob_X_Y[1][k][j] * ... * prob_X_Y[N][k][j]
    # [~Yk] = P(~Yk)          * P(X0=x0|~Yk)         * P(X1=x1|~Yk)         * ... * P(XN=xN|~Yk)
    #       = P(~Yk)          * P(X0=item0(j)|~Yk)   * P(X1=item1(j)|~Yk)   * ... * P(XN=itemN(j)|~Yk)   if x0=item0(j), x1=item1(j), ..., xN=itemN(j)
    #       = (1 - prob_Y[k]) * prob_X_notY[0][k][j] * prob_X_notY[1][k][j] * ... * prob_X_notY[N][k][j]
    
    result = '' # result for each row, for each output column k
    
    for row in range(len(tests)): # for each test data row

        YKs = [] # values of [Yk] for each output column k
        nYKs = [] # values of [~Yk] for each output column k
        
        for k in range(output_cols): # for each output column

            prob = prob_Y[k] # Yk
            nprob = 1.0 - prob_Y[k] # ~Yk
            
            for i in range(test_cols): # for each test column

                # find the value of j_index where Xi=item_i(j_index)
                j_index = 0
                for j in range(len(inputSet[i])):
                    if tests[row][i] == inputSet[i][j]:
                        j_index = j
                        break

                # multiply prob_X_Y[0][k][j], prob_X_Y[1][k][j], ...
                prob *= prob_X_Y[i][k][j_index]

            # append to YKs and nYKs
            YKs.append(prob)
            nYKs.append(prob)

        # compare YKs, nYKs and decide TRUE or FALSE
        temp = []
        for k in range(output_cols):
            if YKs[k] >= nYKs[k]: result += '1,' # TRUE if Yk >= ~Yk
            else: result += '0,' # FALSE if Yk < ~Yk
            
        result = result[:len(result)-1] + '\n'
    
    # [5] write to file
    f = open(testOutputFileName, 'w')
    f.write(result)
    f.close()
