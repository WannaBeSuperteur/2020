import deepLearning_GPU
import deepLearning_GPU_helper as helper
import random
import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from keras.models import Model, model_from_json

def isNumber(s):
    try:
        float(s)
        return True
    except:
        return False

# return timestamp as date
def timestamp(s):
    ss = s.split('-')
    t = datetime.datetime(int(ss[0]), int(ss[1]), int(ss[2]), 0, 0)
    return int(t.timestamp / 86400)

# read training/test data from *.csv file
def getDataFromFile(fn, splitter, useSigmoid, type_):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    result = []
    for i in range(len(flines)):
        row = flines[i].split('\n')[0].split(splitter) # each row
        for j in range(len(row)):
            
            if type_ == 0: # if this value is numeric
                if useSigmoid == True: row[j] = helper.sigmoid(float(row[j])) # using sigmoided output value
                else: row[j] = float(row[j]) # using original value
                
            elif type_ == 1: # if this value is a date
                row[j] = timestamp(row[j]) # return the timestamp of data
                
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

    # get data from file using information in input_output_info.txt, in the form of
    # inputFileName trainCol0 trainCol1 ...
    # outputFileName trainCol0 trainCol1 ...
    # testFileName testCol0 testCol1, ...
    f = open('input_output_info.txt', 'r')
    ioInfo = f.readlines()
    f.close()

    # split of line 0 ~ line 2
    inputSplit = ioInfo[0].split('\n')[0].split(' ')
    outputSplit = ioInfo[1].split('\n')[0].split(' ')
    testSplit = ioInfo[2].split('\n')[0].split(' ')

    # file name part
    inputFileName = inputSplit[0] # input train data file
    outputFileName = outputSplit[0] # output train data file
    testFileName = testSplit[0] # test input data file

    # column no (trainCol) part
    inputCols = inputSplit[1:len(inputSplit)]
    outputCols = outputSplit[1:len(outputSplit)]
    testCols = testSplit[1:len(testSplit)]

    # type: 0(numeric), 1(date), 2(text) for each column
    inputCols_type = []
    outputCols_type = []
    testCols_type = []
    
    for i in range(len(inputCols)):
        if 't' in inputCols[i]: # text
            inputCols_type.append(2)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        elif 'd' in inputCols[i]: # date
            inputCols_type.append(1)
            inputCols[i] = int(inputCols[i][:len(inputCols[i])-1])
        else:
            inputCols_type.append(0)
            inputCols[i] = int(inputCols[i])
        
    for i in range(len(outputCols)):
        if 't' in outputCols[i]: # text
            outputCols_type.append(2)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        elif 'd' in outputCols[i]: # date
            outputCols_type.append(1)
            outputCols[i] = int(outputCols[i][:len(outputCols[i])-1])
        else:
            outputCols_type.append(0)
            outputCols[i] = int(outputCols[i])
        
    for i in range(len(testCols)):
        if 't' in testCols[i]: # text
            testCols_type.append(2)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        elif 'd' in testCols[i]: # date
            testCols_type.append(1)
            testCols[i] = int(testCols[i][:len(testCols[i])-1])
        else:
            testCols_type.append(0)
            testCols[i] = int(testCols[i])

    # read files
    inputs = getDataFromFile(inputFileName, ',', False, inputCols_type) # input train data
    outputs = getDataFromFile(outputFileName, ',', True, outputCols_type) # output train data (using Sigmoid)
    tests = getDataFromFile(testFileName, ',', False, testCols_type) # test input data

    np.set_printoptions(precision=4, linewidth=150)

    # find average and stddev (standard deviation) of each column
    input_avgs = [] # average of each column of train input data
    input_stddevs = [] # standard deviation of each column of train input data
    output_avgs = [] # average of each column of train output data
    output_stddevs = [] # standard deviation of each column of train output data

    # for input and output
    for i in range(len(inputCols)):
        if inputCols_type[i] == 0 or inputCols_type[i] == 1: # numeric or date
            input_avgs.append(np.mean(inputs[:][inputCols[i]], axis=0)[0])
            input_stddevs.append(np.std(inputs[:][inputCols[i]], axis=0)[0])
        else: # text
            input_avgs.append(None)
            input_stddevs.append(None)

        if outputCols_type[i] == 0 or outputCols_type[i] == 1: # numeric or date
            output_avgs.append(np.mean(outputs[:][outputCols[i]], axis=0)[0])
            output_stddevs.append(np.std(outputs[:][outputCols[i]], axis=0)[0])
        else: # text
            output_avgs.append(None)
            output_stddevs.append(None)

    # training and test inputs and outputs
    trainI = [] # train input
    trainO = [] # train output
    testI = [] # test input
    onehotList = [] # one-hot column list of test input, in the form of [column in original data, set of values]
    
    for i in range(len(inputs)):

        # append to trainI (train input)
        trainI_temp = []
        
        for j in range(len(inputCols)):
            if inputCols_type[j] == 0: # just append this value if numeric
                trainI_temp.append(inputs[i][inputCols[j]])

            elif inputCols_type[j] == 1: # date
                trainI_temp.append((inputs[i][inputCols[j]] - input_avgs[j])/input_stddevs[j])
            
            else: # one-hot input (0 or 1) based on memset
                memset = makeSet(inputs, inputCols[j]) # set list of members of this column
                for k in range(len(memset)):
                    if inputs[i][inputCols[j]] == memset[k]: trainI_temp.append(1)
                    else: trainI_temp.append(0)
            
        trainI.append(trainI_temp)

        # append to trainO (train output)
        trainO_temp = []

        for j in range(len(outputCols)):
            if outputCols_type[j] == 0: # just append this value if numeric
                trainO_temp.append(outputs[i][outputCols[j]])

            elif outputCols_type[j] == 1: # date
                trainO_temp.append((outputs[i][outputCols[j]] - output_avgs[j])/output_stddevs[j])
            
            else: # one-hot input (0 or 1) based on memset
                memset = makeSet(outputs, outputCols[j]) # set list of members of this column
                for k in range(len(memset)):
                    if outputs[i][outputCols[j]] == memset[k]: trainO_temp.append(1)
                    else: trainO_temp.append(0)

                if i == 0: onehotList.append([outputCols[j], memset]) # save onehotList
                    
        trainO.append(trainO_temp)

    for i in range(len(tests)):
        
        # append to testI (test input)
        testI_temp = []
        
        for j in range(len(testCols)):
            if testCols_type[j] == 0: # just append this value if numeric
                testI_temp.append(tests[i][testCols[j]])

            elif testCols_type[j] == 1: # date
                testI_temp.append((tests[i][testCols[j]] - train_avgs[j])/train_stddevs[j])
            
            else: # one-hot input (0 or 1) based on memset
                memset = makeSet(tests, testsCols[j]) # set list of members of this column
                for k in range(len(memset)):
                    if tests[i][testCols[j]] == memset[k]: testI_temp.append(1)
                    else: testI_temp.append(0)
                    
        testI.append(testI_temp)

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
    print('\n ---- input data ----\n')
    for i in range(len(trainI)): print(trainI[i])

    print('\n ---- output data ----\n')
    for i in range(len(trainO)): print(trainO[i])

    print('\n ---- test data ----\n')
    for i in range(len(testI)): print(testI[i])

    # learning
    print('\n <<<< LEARNING >>>>\n')
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, 'test', 500, True, True, deviceName)

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
        
        while testIndex < len(outputLayer[0]):

            if originalIndex == onehotList[onehotListIndex][0]: # next item to see in onehotList is column j -> apply one-hot
                memset = onehotList[onehotListIndex][1] # set of members in column j
                #print(str(testIndex) + ' memset:' + str(memset))
                
                # find max index in onehotList
                maxIndexInMemset = 0 # index with maximum value in memset
                maxValueInMemset = -1000 # maximum value in memset
                
                for j in range(len(memset)):
                    if outputLayer[i][testIndex+j] > maxValueInMemset:
                        maxValueInMemset = outputLayer[i][testIndex+j]
                        maxIndexInMemset = j

                # append to result
                result += memset[maxIndexInMemset]
                #print('maxIndexInMemset:' + str(maxIndexInMemset))
                onehotListIndex += 1
                testIndex += len(memset)-1
                
            else: # otherwise just write
                result += str(outputLayer[i][testIndex])

            originalIndex += 1
            testIndex += 1
            if testIndex < len(outputLayer[0])-1: result += ','

            # break when all items in onehotList are found
            if onehotListIndex == len(onehotList): break
            
        result += '\n'

    f = open('test_result.csv', 'w')
    f.write(result)
    f.close()
