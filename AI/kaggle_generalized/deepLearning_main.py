import deepLearning_GPU
import deepLearning_GPU_helper as helper
import random
import tensorflow as tf
import numpy as np
import time
import math
from tensorflow import keras
from keras.models import Model, model_from_json
from datetime import datetime

def isNumber(s):
    try:
        float(s)
        return True
    except:
        return False

# print rounded array
def roundedArray(array, n):
    roundedArray = []
    for i in range(len(array)):
        try:
            roundedArray.append(round(array[i], n))
        except:
            roundedArray.append(array[i])
    return roundedArray

# return timestamp as date
def timestamp_type0(s):
    ts = time.mktime(datetime.strptime(s, '%Y-%m-%d').timetuple())
    return int(ts / 86400)
def timestamp_type1(s):
    ts = time.mktime(datetime.strptime(s, '%m/%d/%Y').timetuple())
    return int(ts / 86400)

# read training/test data from *.csv file
# cols_: list of columns designated as training/test data, e.g. [0, 1, 3, 5, 6, 10]
# type_: type of each column of training/test data, e.g. [0, 1, 2, 0, 0, 1]
def getDataFromFile(fn, splitter, cols_, type_):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    result = []
    for i in range(1, len(flines)):

        # replace all ','s within "", to '#'
        if '"' in flines[i]:
            quoteCount = 0
            for j in range(len(flines[i])):
                if flines[i][j] == '"': quoteCount += 1
                elif flines[i][j] == ',' and quoteCount % 2 == 1: flines[i] = flines[i][:j] + '#' + flines[i][j+1:]

        # parse each row
        row = flines[i].split('\n')[0].split(splitter)

        # for each column designated as training/test data
        for j in range(len(cols_)):

            thisColIndex = cols_[j] # index of this column in the row

            # if this value is numeric
            if type_[j] == 0 or type_[j] == 3 or type_[j] == 4 or type_[j] == 5 or type_[j] == 6 or type_[j] == 7:

                if type_[j] == 3 or type_[j] == 5: # using log
                    row[thisColIndex] = math.log(float(row[thisColIndex]), 2) # using log-ed value
                elif type_[j] == 6 or type_[j] == 7: # using log(x+1)
                    row[thisColIndex] = math.log(float(row[thisColIndex])+1.0, 2) # using log(x+1)-ed value
                else: # not using log
                    row[thisColIndex] = float(row[thisColIndex]) # using original value

            # if this value is a date
            elif type_[j] == 1:
                row[thisColIndex] = timestamp_type0(row[thisColIndex]) # return the timestamp of data (yyyy-mm-dd)
            elif type_[j] == 8:
                row[thisColIndex] = timestamp_type1(row[thisColIndex]) # return the timestamp of data (mm/dd/yyyy)
                
        result.append(row)

    return result

# return set of members of column colNum in the array
# example: array=[['a', 1], ['b', 1], ['c', 2], ['c', 3], ['a', 0]], colNum=0 -> ['a', 'b', 'c']
def makeSet(array, colNum):
    arraySet = []
    for i in range(len(array)): arraySet.append(array[i][colNum])
    arraySet = set(arraySet)
    arraySet = list(arraySet)

    return arraySet

# sign of value
def sign(val):
    if val > 0: return 1
    elif val == 0: return 0
    else: return -1

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # get data from file using information in input_output_info.txt, in the form of
    # inputFileName trainCol0 trainCol1 ...
    # outputFileName trainCol0 trainCol1 ...
    # testFileName testCol0 testCol1 ...
    # testOutputFileName (o or r) (o or r) ...
    f = open('input_output_info.txt', 'r')
    ioInfo = f.readlines()
    f.close()

    # split of line 0 ~ line 3
    inputSplit = ioInfo[0].split('\n')[0].split(' ')
    outputSplit = ioInfo[1].split('\n')[0].split(' ')
    testSplit = ioInfo[2].split('\n')[0].split(' ')
    testOutputSplit = ioInfo[3].split('\n')[0].split(' ')

    # file name part
    inputFileName = inputSplit[0] # input train data file
    outputFileName = outputSplit[0] # output train data file
    testFileName = testSplit[0] # test input data file
    testOutputFileName = testOutputSplit[0] # test output data file

    # column no (trainCol) part
    inputCols = inputSplit[1:len(inputSplit)]
    outputCols = outputSplit[1:len(outputSplit)]
    testCols = testSplit[1:len(testSplit)]
    testOutputCols = testOutputSplit[1:len(testOutputSplit)]

    # type: 0(numeric), 1(date-Zvalue), 2(text), 3(numeric-log), 4(numeric-Zvalue), 5(numeric-logZvalue) for each column
    inputCols_type = []
    outputCols_type = []
    testCols_type = []
    testOutputCols_type = []

    # for train input data
    for i in range(len(inputCols)):
        if 't' in inputCols[i]: # text
            inputCols_type.append(2)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        elif 'd1' in inputCols[i]: # date type mm/dd/yyyy (converted to Z value)
            inputCols_type.append(8)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-2])
        elif 'd' in inputCols[i]: # date type yyyy-mm-dd (converted to Z value)
            inputCols_type.append(1)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        elif 'lpz' in inputCols[i]: # numeric -> Z(log2(numeric+1))
            inputCols_type.append(7)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-3])
        elif 'lp' in inputCols[i]: # numeric -> log2(numeric+1)
            inputCols_type.append(6)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-2])
        elif 'lz' in inputCols[i]: # numeric -> Z(log2(numeric))
            inputCols_type.append(5)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-2])
        elif 'l' in inputCols[i]: # numeric -> log2(numeric)
            inputCols_type.append(3)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        elif 'z' in inputCols[i]: # numeric -> Z(numeric)
            inputCols_type.append(4)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        else: # just numeric
            inputCols_type.append(0)
            inputCols[i] = int(inputCols[i])

    # for train output data
    for i in range(len(outputCols)):
        if 't' in outputCols[i]: # text
            outputCols_type.append(2)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        elif 'd1' in outputCols[i]: # date type mm/dd/yyyy (converted to Z value)
            outputCols_type.append(8)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-2])
        elif 'd' in outputCols[i]: # date type yyyy-mm-dd (converted to Z value)
            outputCols_type.append(1)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        elif 'lpz' in outputCols[i]: # numeric -> Z(log2(numeric+1))
            outputCols_type.append(7)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-3])
        elif 'lp' in outputCols[i]: # numeric -> log2(numeric+1)
            outputCols_type.append(6)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-2])
        elif 'lz' in outputCols[i]: # numeric -> Z(log2(numeric))
            outputCols_type.append(5)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-2])
        elif 'l' in outputCols[i]: # numeric -> log2(numeric)
            outputCols_type.append(3)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        elif 'z' in outputCols[i]: # numeric -> Z(numeric)
            outputCols_type.append(4)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        else: # just numeric
            outputCols_type.append(0)
            outputCols[i] = int(outputCols[i])

    # for test input data
    for i in range(len(testCols)):
        if 't' in testCols[i]: # text
            testCols_type.append(2)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        elif 'd1' in testCols[i]: # date type mm/dd/yyyy (converted to Z value)
            testCols_type.append(8)
            testCols[i] = int(testCols[i][:len(testCols[i])-2])
        elif 'd' in testCols[i]: # date type yyyy-mm-dd (converted to Z value)
            testCols_type.append(1)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        elif 'lpz' in testCols[i]: # numeric -> Z(log2(numeric+1))
            testCols_type.append(7)
            testCols[i] = int(testCols[i][:len(testCols[i])-3])
        elif 'lp' in testCols[i]: # numeric -> log2(numeric+1)
            testCols_type.append(6)
            testCols[i] = int(testCols[i][:len(testCols[i])-2])
        elif 'lz' in testCols[i]: # numeric -> Z(log2(numeric))
            testCols_type.append(5)
            testCols[i] = int(testCols[i][:len(testCols[i])-2])
        elif 'l' in testCols[i]: # numeric -> log2(numeric)
            testCols_type.append(3)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        elif 'z' in testCols[i]: # numeric -> Z(numeric)
            testCols_type.append(4)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        else: # just numeric
            testCols_type.append(0)
            testCols[i] = int(testCols[i])

    # for test output data
    for i in range(len(testOutputCols)):
        if testOutputCols[i] == 'o': testOutputCols_type.append(0)
        else: testOutputCols_type.append(1)

    # read files
    inputs = getDataFromFile(inputFileName, ',', inputCols, inputCols_type) # input train data
    outputs = getDataFromFile(outputFileName, ',', outputCols, outputCols_type) # output train data (using Sigmoid)
    tests = getDataFromFile(testFileName, ',', testCols, testCols_type) # test input data

    # print input, output, and test data
    if printed != 0:
        print('\n ---- original input data ----\n')
        for i in range(len(inputs)): print(roundedArray(inputs[i], 6))

        print('\n ---- original output data ----\n')
        for i in range(len(outputs)): print(roundedArray(outputs[i], 6))

        print('\n ---- original test data ----\n')
        for i in range(len(tests)): print(roundedArray(tests[i], 6))

    np.set_printoptions(precision=4, linewidth=150)

    # find average and stddev (standard deviation) of each column
    input_avgs = [] # average of each column of train input data
    input_stddevs = [] # standard deviation of each column of train input data
    output_avgs = [] # average of each column of train output data
    output_stddevs = [] # standard deviation of each column of train output data

    # for train input
    for i in range(len(inputCols)):

        # for train input data
        if inputCols_type[i] == 2: # text (type == 2)
            input_avgs.append(None)
            input_stddevs.append(None)

        else: # numeric or date (that is, type is not 2)

            # train output column inputCols[i]
            inputCol = []
            for j in range(len(inputs)): inputCol.append(float(inputs[j][inputCols[i]])) # add original value

            # append to _avgs and _stddevs
            input_avgs.append(np.mean(inputCol))
            input_stddevs.append(np.std(inputCol))

    # for train output
    for i in range(len(outputCols)):
        
        # for train input data
        if outputCols_type[i] == 2: # text (type == 2)
            output_avgs.append(None)
            output_stddevs.append(None)

        else: # numeric or date (that is, type is not 2)

            # train output column outputCols[i]
            outputCol = []
            for j in range(len(outputs)): outputCol.append(float(outputs[j][outputCols[i]])) # add original value

            # append to _avgs and _stddevs
            output_avgs.append(np.mean(outputCol))
            output_stddevs.append(np.std(outputCol))

    # training and test inputs and outputs
    trainI = [] # train input
    trainO = [] # train output
    testI = [] # test input
    onehotList = [] # one-hot column list of test input, in the form of [column in the output data, set of values]
    memset = []

    if printed != 0:
        print('')
        print('input_avgs: ' + str(roundedArray(input_avgs, 6)))
        print('input_stds: ' + str(roundedArray(input_stddevs, 6)))
        print('output_avgs: ' + str(roundedArray(output_avgs, 6)))
        print('output_stds: ' + str(roundedArray(output_stddevs, 6)))
    
    for i in range(len(inputs)):

        # append to trainI (train input)
        trainI_temp = []
        memsetIndex = 0 # index for memset
        
        for j in range(len(inputCols)):

            # just append this value if numeric
            if inputCols_type[j] == 0 or inputCols_type[j] == 3 or inputCols_type[j] == 6:
                trainI_temp.append(inputs[i][inputCols[j]])

            # (type:1) Z(numeric) date (yyyy-mm-dd) / (type:8) Z(numeric) date (mm/dd/yyyy)
            # (type:4) Z(log2(numeric)) / (type:5) (Z using log-ed average and stddev) / (type:7) (Z using (log+1)-ed average and stddev)
            elif inputCols_type[j] == 1 or inputCols_type[j] == 4 or inputCols_type[j] == 5 or inputCols_type[j] == 7 or inputCols_type[j] == 8:
                trainI_temp.append((inputs[i][inputCols[j]] - input_avgs[j])/input_stddevs[j])

            # one-hot input (0 or 1) based on memset
            elif inputCols_type[j] == 2:
                if i == 0: memset.append(makeSet(inputs, inputCols[j])) # set list of members of this column
                
                for k in range(len(memset[memsetIndex])):
                    if inputs[i][inputCols[j]] == memset[memsetIndex][k]: trainI_temp.append(1)
                    else: trainI_temp.append(0)

                memsetIndex += 1
            
        trainI.append(trainI_temp)

    memset = [] # initialize memset

    for i in range(len(outputs)):

        # append to trainO (train output)
        trainO_temp = []
        memsetIndex = 0 # index for memset

        for j in range(len(outputCols)):
            
            # just append this value if numeric
            if outputCols_type[j] == 0 or outputCols_type[j] == 3 or outputCols_type[j] == 6:
                trainO_temp.append(outputs[i][outputCols[j]])

            # (type:1) Z(numeric) date (yyyy-mm-dd) / (type:8) Z(numeric) date (mm/dd/yyyy)
            # (type:4) Z(log2(numeric)) / (type:5) (Z using log-ed average and stddev) / (type:7) (Z using (log+1)-ed average and stddev)
            elif outputCols_type[j] == 1 or outputCols_type[j] == 4 or outputCols_type[j] == 5 or outputCols_type[j] == 7 or outputCols_type[j] == 8:
                trainO_temp.append((outputs[i][outputCols[j]] - output_avgs[j])/output_stddevs[j])

            # one-hot input (0 or 1) based on memset
            elif outputCols_type[j] == 2:
                if i == 0: memset.append(makeSet(outputs, outputCols[j])) # set list of members of this column
                
                for k in range(len(memset[memsetIndex])):
                    if outputs[i][outputCols[j]] == memset[memsetIndex][k]: trainO_temp.append(1)
                    else: trainO_temp.append(0)

                if i == 0: onehotList.append([j, memset[memsetIndex]]) # save onehotList

                memsetIndex += 1

        # apply sigmoid to each value of trainO
        for k in range(len(trainO_temp)): trainO_temp[k] = helper.sigmoid(trainO_temp[k])
                    
        trainO.append(trainO_temp)

    memset = [] # initialize memset

    for i in range(len(tests)):
        
        # append to testI (test input)
        testI_temp = []
        memsetIndex = 0 # index for memset
        
        for j in range(len(testCols)):
            
            # just append this value if numeric
            if testCols_type[j] == 0 or testCols_type[j] == 3 or testCols_type[j] == 6:
                testI_temp.append(tests[i][testCols[j]])

            # (type:1) Z(numeric) date (yyyy-mm-dd) / (type:8) Z(numeric) date (mm/dd/yyyy)
            # (type:4) Z(log2(numeric)) / (type:5) (Z using log-ed average and stddev) / (type:7) (Z using (log+1)-ed average and stddev)
            elif testCols_type[j] == 1 or testCols_type[j] == 4 or testCols_type[j] == 5 or testCols_type[j] == 7 or outputCols_type[j] == 8:
                testI_temp.append((tests[i][testCols[j]] - input_avgs[j])/input_stddevs[j])

            # one-hot input (0 or 1) based on memset
            elif testCols_type[j] == 2:
                if i == 0: memset.append(makeSet(tests, testCols[j])) # set list of members of this column
                
                for k in range(len(memset[memsetIndex])):
                    if tests[i][testCols[j]] == memset[memsetIndex][k]: testI_temp.append(1)
                    else: testI_temp.append(0)

                memsetIndex += 1
                    
        testI.append(testI_temp)

    print(onehotList)

    # model design using deepLearning_model.txt, in the form of

    ## layers
    # FI                     (tf.keras.layers.Flatten(input_shape=(len(trainI[0]),)))
    # F                      (keras.layers.Flatten())
    # D 16 relu              (keras.layers.Dense(16, activation='relu'))
    # DO sigmoid             (keras.layers.Dense(len(trainO[0]), activation='sigmoid'))
    # Drop 0.25              (keras.layers.Dropout(0.25))
    # C2DI 32 3 3 12 12 relu (keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(12, 12, 1), activation='relu'))
    # C2D 32 3 3 relu        (keras.layers.Conv2D(32, (3, 3), activation='relu'))
    # MP 2                   (keras.layers.MaxPooling2D(pool_size=2))
    # R 12 12                (tf.keras.layers.Reshape((12, 12, 1), input_shape=(12*12,))

    ## optimizers
    # OP adadelta 0.001 0.95 1e-07    (tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07))
    # OP adagrad 0.001 0.1 1e-07      (tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07))
    # OP adam0 0.001                  (tf.keras.optimizers.Adam(0.001))
    # OP adam1 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False))
    # OP adamax 0.001 0.9 0.999 1e-07 (tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
    # OP nadam 0.001 0.9 0.999 1e-07  (tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))
    # OP rmsprop 0.001 0.9 0.0 1e-07  (tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07))
    # OP sgd 0.01 0.0                 (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False))

    # activation function of final layer is always 'sigmoid'
    
    f = open('deepLearning_model.txt', 'r')
    modelInfo = f.readlines()
    f.close()

    # Neural Network
    NN = []
    for i in range(len(modelInfo)):
        info = modelInfo[i].split('\n')[0]
        infoSplit = info.split(' ')

        # add layers to Neural Network as below
        if info == 'FI': NN.append(tf.keras.layers.Flatten(input_shape=(len(trainI[0]),)))
        elif info == 'F': NN.append(keras.layers.Flatten())
        elif infoSplit[0] == 'D': NN.append(keras.layers.Dense(int(infoSplit[1]), activation=infoSplit[2]))
        elif infoSplit[0] == 'DO': NN.append(keras.layers.Dense(len(trainO[0]), activation=infoSplit[1]))
        elif infoSplit[0] == 'Drop': NN.append(keras.layers.Dropout(float(infoSplit[1])))
        elif infoSplit[0] == 'C2DI':
            NN.append(keras.layers.Conv2D(int(infoSplit[1]), kernel_size=(int(infoSplit[2]), int(infoSplit[3])),
                                          input_shape=(int(infoSplit[4]), int(infoSplit[5]), 1), activation=infoSplit[6]))
        elif infoSplit[0] == 'C2D':
            NN.append(keras.layers.Conv2D(int(infoSplit[1]), (int(infoSplit[2]), int(infoSplit[3])),
                                          activation=infoSplit[4]))
        elif infoSplit[0] == 'MP':
            NN.append(keras.layers.MaxPooling2D(pool_size=int(infoSplit[1])))
        elif infoSplit[0] == 'R':
            NN.append(tf.keras.layers.Reshape((int(infoSplit[1]), int(infoSplit[2]), 1),
                                              input_shape=(int(infoSplit[1])*int(infoSplit[2]),)))

    # optimizer
    op = None
    for i in range(len(modelInfo)):
        info = modelInfo[i].split('\n')[0]
        infoSplit = info.split(' ')

        # specify optimizer
        if infoSplit[0] == 'OP':
            if infoSplit[1] == 'adadelta': op = tf.keras.optimizers.Adadelta(learning_rate=float(infoSplit[2]),
                                                                             rho=float(infoSplit[3]),
                                                                             epsilon=float(infoSplit[4]))
            
            elif infoSplit[1] == 'adagrad': op = tf.keras.optimizers.Adagrad(learning_rate=float(infoSplit[2]),
                                                                             initial_accumulator_value=float(infoSplit[3]),
                                                                             epsilon=float(infoSplit[4]))
            
            elif infoSplit[1] == 'adam0': op = tf.keras.optimizers.Adam(float(infoSplit[2]))
            
            elif infoSplit[1] == 'adam1': op = tf.keras.optimizers.Adam(learning_rate=float(infoSplit[2]),
                                                                        beta_1=float(infoSplit[3]),
                                                                        beta_2=float(infoSplit[4]),
                                                                        epsilon=float(infoSplit[5]), amsgrad=False)
            
            elif infoSplit[1] == 'adamax': op = tf.keras.optimizers.Adamax(learning_rate=float(infoSplit[2]),
                                                                           beta_1=float(infoSplit[3]),
                                                                           beta_2=float(infoSplit[4]),
                                                                           epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'nadam': op = tf.keras.optimizers.Nadam(learning_rate=float(infoSplit[2]),
                                                                         beta_1=float(infoSplit[3]),
                                                                         beta_2=float(infoSplit[4]),
                                                                         epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'rmsprop': op = tf.keras.optimizers.RMSprop(learning_rate=float(infoSplit[2]),
                                                                             rho=float(infoSplit[3]),
                                                                             momentum=float(infoSplit[4]),
                                                                             epsilon=float(infoSplit[5]))
            
            elif infoSplit[1] == 'sgd': op = tf.keras.optimizers.SGD(learning_rate=float(infoSplit[2]),
                                                                     momentum=float(infoSplit[3]), nesterov=False)
            
            break

    # print input, output, and test data
    if printed != 0:
        print('\n ---- input data ----\n')
        for i in range(len(trainI)): print(roundedArray(trainI[i], 6))

        print('\n ---- output data ----\n')
        for i in range(len(trainO)): print(roundedArray(trainO[i], 6))

        print('\n ---- test data ----\n')
        for i in range(len(testI)): print(roundedArray(testI[i], 6))

    # learning
    print('\n <<<< LEARNING >>>>\n')
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, 'test', epoch, False, True, deviceName)

    # test
    print('\n <<<< TEST >>>>\n')
        
    print('\n << test output (첫번째 Neural Network) >>\n')
    newModel = deepLearning_GPU.deepLearningModel('test', True)
    testOutput = deepLearning_GPU.modelOutput(newModel, testI)
    print('\ntest output에 대한 테스트 결과:\ntest_result.csv')
    print('\none-hot list:\n' + str(onehotList))

    # estimate
    outputLayer = testOutput[len(testOutput)-1]

    # inverse sigmoid
    for i in range(len(outputLayer)): # for each output data
        for j in range(len(outputLayer[0])): # for each value of output data
            outputLayer[i][j] = helper.invSigmoid(outputLayer[i][j])

    # write to file
    result = ''
    print('\n<<<< output layer >>>>')
    
    for i in range(len(outputLayer)):
        #print('output layer ' + str(i) + ' : ' + str(outputLayer[i]))
        
        originalIndex = 0 # column index of original test data (one-hot not applied)
        onehotListIndex = 0 # next index to see in onehotList
        testIndex = 0 # column index of test data (one-hot applied)

        onehotList.append([None]) # append None to prevent index error
        
        while testIndex < len(outputLayer[0]):

            # round or not (round if column type is 1)
            rounded = False
            if testOutputCols_type[originalIndex] == 1:
                rounded = True

            # next item to see in onehotList is column j -> apply one-hot (text or text-like number)
            if originalIndex == onehotList[onehotListIndex][0]:
                memset = onehotList[onehotListIndex][1] # set of members in column j
                
                # find max index in onehotList
                maxIndexInMemset = 0 # index with maximum value in memset
                maxValueInMemset = -1000 # maximum value in memset
                
                for j in range(len(memset)):
                    if outputLayer[i][testIndex+j] > maxValueInMemset:
                        maxValueInMemset = outputLayer[i][testIndex+j]
                        maxIndexInMemset = j

                # append to result
                result += memset[maxIndexInMemset]
                onehotListIndex += 1
                testIndex += len(memset)-1

            # date
            elif outputCols_type[originalIndex] == 1:
                outputDate = outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex]
                date = datetime.fromtimestamp(outputDate * 86400)
                result += date.strftime('%Y-%m-%d')
            elif outputCols_type[originalIndex] == 8:
                outputDate = outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex]
                date = datetime.fromtimestamp(outputDate * 86400)
                result += date.strftime('%m/%d/%Y')

            # numeric value
            elif outputCols_type[originalIndex] == 0:
                if rounded == True: result += str(round(outputLayer[i][testIndex], 0))
                else: result += str(outputLayer[i][testIndex])

            # log2(numeric) -> 2^(numeric)
            elif outputCols_type[originalIndex] == 3:
                if rounded == True: result += str(round(pow(2.0, outputLayer[i][testIndex]), 0))
                else: result += str(pow(2.0, outputLayer[i][testIndex]))

            # Z(numeric) -> recover
            elif outputCols_type[originalIndex] == 4:
                if rounded == True: result += str(round(outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex], 0))
                else: result += str(outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex])

            # Z(log2(numeric)) -> recover(un-Z using log-ed average and stddev) -> 2^(recover)
            elif outputCols_type[originalIndex] == 5:
                recover = outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex]
                
                if rounded == True: result += str(round(pow(2.0, recover), 0))
                else: result += str(pow(2.0, recover))

            # log2(numeric+1) -> 2^(numeric)
            elif outputCols_type[originalIndex] == 6:
                if rounded == True: result += str(round(pow(2.0, outputLayer[i][testIndex])-1.0, 0))
                else: result += str(pow(2.0, outputLayer[i][testIndex])-1.0)

            # Z(log2(numeric+1)) -> recover(un-Z using log-ed average and stddev)+1 -> 2^(recover)
            elif outputCols_type[originalIndex] == 7:
                recover = outputLayer[i][testIndex] * output_stddevs[originalIndex] + output_avgs[originalIndex]

                if rounded == True: result += str(round(pow(2.0, recover)-1.0, 0))
                else: result += str(pow(2.0, recover)-1.0)

            originalIndex += 1
            testIndex += 1

            # break when all items are found
            if testIndex < len(outputLayer[0]): result += ','
            else: break
            
        result += '\n'

    f = open(testOutputFileName, 'w')
    f.write(result)
    f.close()
