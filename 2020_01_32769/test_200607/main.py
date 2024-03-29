import deepLearning_GPU
import deepLearning_GPU_helper as h
import tensorflow as tf
import random
from tensorflow import keras
from keras.models import Model, model_from_json

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')

    all_resultText = ''
    all_resultTable = 'epoch\tscaling\t#test\tdifAvg\t0difAvg\taccur\tTP\tTN\tFP\tFN\tP\tN\tcorrect\nnote: difAvg error (squared) and 0-difAvg error (squared) are multiplied by 100\n'

    f = open('info.txt', 'r')
    fr = f.readlines()
    f.close()
    lenInput = int(fr[1].split(' ')[8]) # number of columns in each input data

    inf = 2147483647

    # learning and test
    for epoch in [50]:
        for scalingRate in [4]:

            # training and test data
            ftr = open('paper_train.txt', 'r')
            ftrr = ftr.readlines()
            ftr.close()

            fte = open('paper_test.txt', 'r')
            fter = fte.readlines()
            fte.close()

            trainInputs = [] # input data for training
            trainOutputs = [] # output data for training
            testInputs = [] # input data for test
            testValOutputs = [] # output data for test (for validation)
            
            for i in range(len(ftrr)):
                ftrrs = ftrr[i].split('\t')
                for j in range(len(ftrrs)): ftrrs[j] = float(ftrrs[j])
                trainInputs.append(ftrrs[0:lenInput])
                trainOutputs.append([ftrrs[lenInput]])

            for i in range(len(fter)):
                fters = fter[i].split('\t')
                for j in range(len(fters)): fters[j] = float(fters[j])
                testInputs.append(fters[0:lenInput])
                testValOutputs.append(fters[lenInput])

            # scaling trainOutputs and testValOutputs (value' <- value * scalingRate - 1.0)
            for i in range(len(trainOutputs)):
                trainOutputs[i][0] = trainOutputs[i][0] * scalingRate - 1.0

                # consider math range error for sigmoid
                if scalingRate < inf:
                    trainOutputs[i][0] = h.sigmoid(trainOutputs[i][0])
                else:
                    trainOutputs[i][0] = 1.0
                    
            for i in range(len(testValOutputs)):
                testValOutputs[i] = testValOutputs[i] * scalingRate - 1.0

            # print training data
            print('\n <<< training data >>>\n')
            print('TRAIN length: ' + str(len(trainInputs)))
            for i in range(min(100, len(trainInputs))):
                print(str(i) + ' : ' + str(trainInputs[i]))
                print(' -> ' + str(trainOutputs[i]))
            if len(trainInputs) > 100: print('...')

            # print test data
            print('\n <<< test data >>>\n')
            print('TEST length: ' + str(len(testInputs)))
            for i in range(min(100, len(testInputs))): print(str(i) + ' : ' + str(testInputs[i]))
            if len(testInputs) > 100: print('...')
        
            # learning
            print('\n <<< LEARNING >>>\n')
            
            NN = [tf.keras.layers.Flatten(input_shape=(lenInput,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')]
            op = tf.keras.optimizers.Adam(0.0005) # learning rate = 0.0005
            
            deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', trainInputs, trainOutputs, 'mainCOVID19_' + str(epoch), epoch, False, True, deviceName)

            # test
            print('\n <<< TEST >>>\n')
            newModel = deepLearning_GPU.deepLearningModel('mainCOVID19_' + str(epoch), True)
            testOutputs = deepLearning_GPU.modelOutput(newModel, testInputs)

            # print output layer of test result
            print('\n <<< test result >>>\n')
            difSum = 0 # sum of difference (absolute)
            valOutputSum = 0 # sum of validation output (absolute)
            
            result = testOutputs[len(testOutputs)-1]

            binary = [0, 0, 0, 0] # True-positive, True-negative, False-positive, False-negative
            
            for i in range(len(result)):
                result[i] = h.invSigmoid(result[i], inf) # invSigmoid results

                # recover results and testValOutputs
                # value * scalingRate - 1.0 <- value' <==> value <- (value' + 1.0) / scalingRate
                result[i] = (result[i] + 1.0) / scalingRate
                testValOutputs[i] = (testValOutputs[i] + 1.0) / scalingRate

                # calculate difference
                dif = pow(result[i]-testValOutputs[i], 2) # using mean-squared error
                print('test out: ' + str(result[i]) + ' val: ' + str(testValOutputs[i]) + ' -> dif: ' + str(dif))
                
                difSum += float(dif)
                valOutputSum += pow(testValOutputs[i], 2) # using mean-squared error

                if result[i] >= 0 and testValOutputs[i] >= 0: binary[0] += 1 # True positive
                elif result[i] < 0 and testValOutputs[i] < 0: binary[1] += 1 # True negative
                elif result[i] >= 0 and testValOutputs[i] < 0: binary[2] += 1 # False positive
                elif result[i] < 0 and testValOutputs[i] >= 0: binary[3] += 1 # False negative

            # print result
            resultText = '\n < epoch = ' + str(epoch) + ' / scalingRate = ' + str(scalingRate) + ' >\ndifference                 -> sum: ' + str(difSum) + ' avg: ' + str(difSum/len(result)) + '\n'
            resultText += 'dif when test out is all 0 -> sum: ' + str(valOutputSum) + ' avg: ' + str(valOutputSum/len(result)) + '\n'
            resultText += 'accuracy rate              -> ' + str(valOutputSum/difSum) + '\n'

            # print each rate (useful only when output is binary value (-1 or 1))
            resultText += 'True positive : ' + str(binary[0]) + ' / ' + str(round(100 * binary[0] / len(result), 2)) + '%\n'
            resultText += 'True negative : ' + str(binary[1]) + ' / ' + str(round(100 * binary[1] / len(result), 2)) + '%\n'
            resultText += 'False positive: ' + str(binary[2]) + ' / ' + str(round(100 * binary[2] / len(result), 2)) + '%\n'
            resultText += 'False negative: ' + str(binary[3]) + ' / ' + str(round(100 * binary[3] / len(result), 2)) + '%\n'

            resultText += 'positive      : ' + str(binary[0]+binary[3]) + ' / ' + str(round(100 * (binary[0]+binary[3]) / len(result), 2)) + '%\n'
            resultText += 'negative      : ' + str(binary[1]+binary[2]) + ' / ' + str(round(100 * (binary[1]+binary[2]) / len(result), 2)) + '%\n'
            resultText += 'correct       : ' + str(binary[0]+binary[1]) + ' / ' + str(round(100 * (binary[0]+binary[1]) / len(result), 2)) + '%\n'

            # save result file
            all_resultText += resultText

            # save result table file
            print(valOutputSum/difSum)
            all_resultTable += str(epoch) + '\t' + str(scalingRate) + '\t' + str(len(result)) + '\t'
            all_resultTable += str(round(100 * difSum/len(result), 4)) + '\t' + str(round(100 * valOutputSum/len(result), 4)) + '\t' + str(round(valOutputSum/difSum, 4)) + '\t'
            all_resultTable += str(round(100 * binary[0] / len(result), 4)) + '\t' + str(round(100 * binary[1] / len(result), 4)) + '\t'
            all_resultTable += str(round(100 * binary[2] / len(result), 4)) + '\t' + str(round(100 * binary[3] / len(result), 4)) + '\t'
            all_resultTable += str(round(100 * (binary[0] + binary[3]) / len(result), 4)) + '\t'
            all_resultTable += str(round(100 * (binary[1] + binary[2]) / len(result), 4)) + '\t'
            all_resultTable += str(round(100 * (binary[0] + binary[1]) / len(result), 4)) + '\n'
            
            f = open('paper_result.txt', 'w')
            f.write(all_resultText)
            f.close()

            f = open('paper_table.txt', 'w')
            f.write(all_resultTable)
            f.close()
            
            print(resultText)
