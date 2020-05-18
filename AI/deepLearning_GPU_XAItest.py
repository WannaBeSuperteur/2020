import deepLearning_GPU
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Model, model_from_json

def getDataFromFile(fn, splitter):
    f = open(fn, 'r')
    flines = f.readlines()
    f.close()

    result = []
    for i in range(len(flines)):
        row = flines[i].split('\n')[0].split(splitter)
        for j in range(len(row)): row[j] = float(row[j])
        result.append(row)

    return result

def sign(val):
    if val > 0: return 1
    elif val == 0: return 0
    else: return -1

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')

    inputs = getDataFromFile('inputs_XAI.txt', '\t')
    outputs = getDataFromFile('outputs_XAI.txt', '\t')

    np.set_printoptions(precision=4, linewidth=150)

    # training and test inputs and outputs
    trainI = []
    trainO = []
    testI = []
    testO = []
    
    for i in range(len(inputs)):
        if random.random() < 0.2:
            testI.append(inputs[i])
            testO.append(outputs[i])
        else:
            trainI.append(inputs[i])
            trainO.append(outputs[i])

    # model design
    NN = [tf.keras.layers.Flatten(input_shape=(len(inputs[0]),)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(len(outputs[0]), activation='sigmoid')]
    op = tf.keras.optimizers.Adam(0.001)

    # learning
    print('\n ' + ('#==='*16) + ' <<<< LEARNING >>>> ' + ('===#'*16) + '\n')
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainI, trainO, 'test', 500, True, True, deviceName)

    # test
    print('\n ' + ('#==='*16) + ' <<<< TEST >>>> ' + ('===#'*16) + '\n')
        
    print('\n << test output (첫번째 Neural Network) >>\n')
    newModel = deepLearning_GPU.deepLearningModel('test', True)
    testOutput = deepLearning_GPU.modelOutput(newModel, testI)
    print('\ntest output에 대한 테스트 결과:\n')

    # estimate
    outputLayer = testOutput[len(testOutput)-1]
    print('')
    print('layers in the network: ' + str(len(testOutput)))
    print('')
    difAbs = 0.0
    
    for i in range(len(testO)):
        op = outputLayer[i] # output layer output
        to = testO[i] # test output
        difA = (abs(op[0]-to[0]) + abs(op[1]-to[1]))/2
        difAbs += difA

        print('output=' + str(np.array(op)) + ' test=' + str(np.array(to)) + ' difAbsoulte=' + str(difA))
    
    print('sum of difAbsoulte=' + str(difAbs) + ', avg of difAbsoulte=' + str(difAbs/len(testO)) + '\n\n')

    # use XAI to find key node for output of AI
    weights = newModel.weights

    R = [[], [], [], [], [], []] # R0, R1, ..., R5, each means R^(0), R^(1), ..., R^(5)
    name = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5'] # name of each layer in R
    neurons = [len(inputs[0]), 16, 16, 16, 16, len(outputs[0])] # number of neurons in each layer

    XAIlist = [] # list of [index, input, output, XAI layer 0]

    # option: basic, evar0(e-variant with e=0.01), evar1(e-variant with e=100),
    #         bvar0(b-variant with a=1,b=0), bvar1(b-variant with a=2,b=-1)
    options = ['basic', 'evar0', 'evar1', 'bvar0', 'bvar1']

    # 30% probability for each test data
    for d in range(len(testO)): # for each test data

        if random.random() > 0.3: continue

        # apply each option
        for option in options:
            
            # for layer 5(output)
            R[5] = [outputLayer[d][0], outputLayer[d][1]]
            
            # for layer 4(hidden)-5(output) ~ layer 0(input)-1(hidden)
            for layer in range(4, -1, -1):
                R[layer] = []
                for i in range(neurons[layer]):
                    temp = 0.0

                    # calculate value by option
                    # 'basic', 'evar0' or 'evar1': do not use zij+ or zij-
                    if option == 'basic' or option == 'evar0' or option == 'evar1':
                        for j in range(neurons[layer+1]):
                            zij = testOutput[layer][d][i] * float(weights[layer*2][i][j])
                            Sumzij = 0.0
                            for k in range(neurons[layer]): Sumzij += testOutput[layer][d][k] * float(weights[layer*2][k][j])

                            # add to temp by option
                            if option == 'basic': temp += zij * R[layer+1][j] / Sumzij
                            elif option == 'evar0': temp += zij * R[layer+1][j] / (Sumzij + 0.01 * sign(Sumzij))
                            elif option == 'evar1': temp += zij * R[layer+1][j] / (Sumzij + 100 * sign(Sumzij))

                    # 'bvar0' or 'bvar1': use zij+ and zij-
                    elif option == 'bvar0' or option == 'bvar1':

                        if option == 'bvar0': # a=1,b=0
                            a = 1.0
                            b = 0.0
                        elif option == 'bvar1': # a=2,b=-1
                            a = 2.0
                            b = -1.0
                        
                        for j in range(neurons[layer+1]):
                            Sumzijp = 0.0
                            Sumzijm = 0.0

                            # find zij (zij+ or zij-)
                            zij = testOutput[layer][d][i] * float(weights[layer*2][i][j])

                            # find Sum(zij+) and Sum(zij-)
                            for k in range(neurons[layer]):
                                zkj = testOutput[layer][d][k] * float(weights[layer*2][k][j])
                                if zkj >= 0: Sumzijp += zkj
                                else: Sumzijm += zkj

                            # add to temp
                            if zij >= 0: temp += a * zij * R[layer+1][j] / Sumzijp # a * zij+ / Sum(zij+)
                            else: temp += b * zij * R[layer+1][j] / Sumzijm # b * zij- / Sum(zij-)
                        
                    R[layer].append(temp)

            # print results
            print('\n\n<<< test data ' + str(d) + ' : ' + str(np.array(testOutput[0][d])) + ' (' + option + ') >>>')
            for layer in range(5, -1, -1):  print(name[layer] + ': ' + str(np.array(R[layer])))

            # divide each element of R[0] by max(abs(R[0]))
            R0_ = []
            for i in range(len(R[0])): R0_.append(R[0][i]/max(max(R[0]), (-1)*min(R[0])))

            # append to XAI array
            XAIlist.append([d, option, testOutput[0][d], R[5], R0_])

    # print XAI array
    print('\n <<< XAI list >>>')

    # influence (heatmap) of each input value for all sampled (each 30% prob) test data for each option
    # based on the sum of XAIlist[i][3] (max value 1)
    influenceOfInput = [[0] * len(inputs[0]) for _ in range(5)]
    
    for i in range(len(XAIlist)):
        print(str(XAIlist[i][0]) + ' -> option=' + str(np.array(XAIlist[i][1])) + ', input=' + str(np.array(XAIlist[i][2])) +
              ', output=' + str(np.array(XAIlist[i][3])) + ', XAI layer0(max:1)=' + str(np.array(XAIlist[i][4])))

        # add influence of each input for this data
        for j in range(len(inputs[0])):
            if np.array(XAIlist[i][1]) == 'basic': influenceOfInput[0][j] += abs(np.array(XAIlist[i][4])[j])
            elif np.array(XAIlist[i][1]) == 'evar0': influenceOfInput[1][j] += abs(np.array(XAIlist[i][4])[j])
            elif np.array(XAIlist[i][1]) == 'evar1': influenceOfInput[2][j] += abs(np.array(XAIlist[i][4])[j])
            elif np.array(XAIlist[i][1]) == 'bvar0': influenceOfInput[3][j] += abs(np.array(XAIlist[i][4])[j])
            elif np.array(XAIlist[i][1]) == 'bvar1': influenceOfInput[4][j] += abs(np.array(XAIlist[i][4])[j])

    print('')
    print('sampled data: ' + str(len(XAIlist)))
    print('influence of input for basic option: ' + str(np.array(influenceOfInput[0])))
    print('influence of input for evar0 option: ' + str(np.array(influenceOfInput[1])))
    print('influence of input for evar1 option: ' + str(np.array(influenceOfInput[2])))
    print('influence of input for bvar0 option: ' + str(np.array(influenceOfInput[3])))
    print('influence of input for bvar1 option: ' + str(np.array(influenceOfInput[4])))
