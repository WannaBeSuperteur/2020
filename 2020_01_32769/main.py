import deepLearning_GPU
import deepLearning_GPU_helper as h
import tensorflow as tf
import random
from tensorflow import keras
from keras.models import Model, model_from_json

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')

    all_resultText = ''
    all_resultTable = 'epoch\tdropout\t#test\tdifAvg\t0difAvg\taccur\tTP\tTN\tFP\tFN\tP\tN\tcorrect\n'

    # learning and test
    for epoch in [5, 15, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000]:
        for dropout in [0, 0, 0, 0, 0]:

            # training and test data
            f = open('paper_train.txt', 'r')
            fr = f.readlines()
            f.close()

            trainInputs = [] # input data for training
            trainOutputs = [] # output data for training
            testInputs = [] # input data for test
            testValOutputs = [] # output data for test (for validation)
            for i in range(len(fr)):
                frs = fr[i].split('\t')
                for j in range(len(frs)): frs[j] = float(frs[j])

                # 80% of data for training and 20% for test
                if random.random() < 0.8:
                    trainInputs.append(frs[1:6])
                    trainOutputs.append([h.sigmoid(frs[6])])
                else:
                    testInputs.append(frs[1:6])
                    testValOutputs.append(frs[6])

            # print training data
            print('\n <<< training data >>>\n')
            print('TRAIN length: ' + str(len(trainInputs)))
            for i in range(min(100, len(trainInputs))): print(str(i) + ' : ' + str(trainInputs[i]) + ' -> ' + str(trainOutputs[i]))
            if len(trainInputs) > 100: print('...')

            # print test data
            print('\n <<< test data >>>\n')
            print('TEST length: ' + str(len(testInputs)))
            for i in range(min(100, len(testInputs))): print(str(i) + ' : ' + str(testInputs[i]))
            if len(testInputs) > 100: print('...')
        
            # learning
            print('\n <<< LEARNING >>>\n')
            
            NN = [tf.keras.layers.Flatten(input_shape=(5,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')]
            op = tf.keras.optimizers.Adam(0.001) # learning rate = 0.001
            
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
                result[i] = h.invSigmoid(result[i], 1000) # invSigmoid results
                
                dif = abs(result[i]-testValOutputs[i])
                print('test out: ' + str(result[i]) + ' val: ' + str(testValOutputs[i]) + ' -> dif: ' + str(dif))
                
                difSum += float(dif)
                valOutputSum += abs(testValOutputs[i])

                if result[i] >= 0 and testValOutputs[i] >= 0: binary[0] += 1 # True positive
                elif result[i] < 0 and testValOutputs[i] < 0: binary[1] += 1 # True negative
                elif result[i] >= 0 and testValOutputs[i] < 0: binary[2] += 1 # False positive
                elif result[i] < 0 and testValOutputs[i] >= 0: binary[3] += 1 # False negative

            # print result
            resultText = '\n < epoch = ' + str(epoch) + ' / drouput = ' + str(dropout) + ' >\ndifference                 -> sum: ' + str(difSum) + ' avg: ' + str(difSum/len(result)) + '\n'
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
            all_resultTable += str(epoch) + '\t' + str(dropout) + '\t' + str(len(result)) + '\t'
            all_resultTable += str(round(difSum/len(result), 4)) + '\t' + str(round(valOutputSum/len(result), 4)) + '\t' + str(round(valOutputSum/difSum, 4)) + '\t'
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
