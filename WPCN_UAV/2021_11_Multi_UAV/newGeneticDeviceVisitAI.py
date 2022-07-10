import throughputTest_NewGenetic as T_NG
import numpy as np
import random
import math
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt

# save input data as image (n: the number of devices)
def saveInputDataImg(input_data_row, row_index, n, swappedMovement, bruteForceMovement):

    movements = [swappedMovement, bruteForceMovement]

    # save image for both swappedMovement and bruteForceMovement
    for i in range(2):
        movement = movements[i]

        plt.clf()
        plt.figure(figsize=(4.0, 4.0))

        if i == 0:
            plt.suptitle('new genetic device visit AI input data (swapped)')
        elif i == 1:
            plt.suptitle('new genetic device visit AI input data (brute-force best)')
            
        plt.axis([-3, 3, -3, 3])

        for j in range(n):
            plt.scatter(input_data_row[2*j], input_data_row[2*j + 1], marker='o', c='blue')

        for j in range(n - 1):
            x = [input_data_row[2*movement[j]    ], input_data_row[2*movement[j+1]    ]]
            y = [input_data_row[2*movement[j] + 1], input_data_row[2*movement[j+1] + 1]]
            
            plt.plot(x, y, linewidth=0.75, c='black')

        plt.savefig('newGenetic_AI_input_' + ('%04d' % (row_index - 1)) + '_' + str(i))

# brute force search for the path
def doBruteForce(deviceList, initialLocUAV, initialMovement):

    devCnt = len(deviceList)
    movements = list(permutations(range(devCnt), devCnt))
    resultDist = 1000000
    resultMovement = list(range(devCnt))

    for movement in movements:
        dist = T_NG.computeTotalDist(initialLocUAV, list(movement), deviceList)
        
        if dist < resultDist:
            resultMovement = list(movement)
            resultDist = dist

    return resultMovement

def test(input_data, output_data, print_input_data):
    
    # randomly place device
    deviceList = []
    n = random.randint(4, 6)
    
    for i in range(n):
        deviceList.append([random.random(), random.random()])

    # normalize device locations
    deviceListNp = np.array(deviceList)

    avg    = np.mean(deviceListNp, axis=0)
    stddev = np.std(deviceListNp, axis=0)

    for i in range(n):
        deviceList[i][0] = (deviceList[i][0] - avg[0]) / stddev[0]
        deviceList[i][1] = (deviceList[i][1] - avg[1]) / stddev[1]

    initialLocUAV = [0, 0, 0]
    initialMovement = list(range(n))

    # test total distance
    swappedMovement    = T_NG.swapBasic(deviceList, initialLocUAV, initialMovement, False)
    bruteForceMovement = doBruteForce(deviceList, initialLocUAV, initialMovement)

    # compute total distance
    totalDistSwapped    = T_NG.computeTotalDist(initialLocUAV, swappedMovement   , deviceList)
    totalDistBruteForce = T_NG.computeTotalDist(initialLocUAV, bruteForceMovement, deviceList)

    # write input and output data
    input_d = []
    
    for i in range(n):
        if i == 0:
            input_d.append(deviceList[0][0])
            input_d.append(deviceList[0][1])
        else:
            input_d.append(deviceList[i][0] - deviceList[i-1][0])
            input_d.append(deviceList[i][1] - deviceList[i-1][1])

    # fill the blank cells
    for i in range(2 * (6 - n)):
        input_d.append(0.0)

    # append to input_data and output_data
    input_data .append(input_d)
    output_data.append([math.log(totalDistSwapped / totalDistBruteForce, 2.0)])

    if print_input_data == True:
        print('input data :', np.round_(input_data[-1], 4), 'dist :', round(totalDistSwapped, 4), round(totalDistBruteForce, 4))

        # save input data as image
        saveInputDataImg(input_data[-1], len(input_data), n, swappedMovement, bruteForceMovement)

def defineModel(train_input, train_output, test_input, test_output, epochs):
    
    model = GENETIC_VISIT_AI_MODEL()

    # training setting (early stopping and reduce learning rate)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             mode="min",
                                             patience=5)

    lr_reduced = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=2,
                                                      verbose=1,
                                                      min_delta=0.0001,
                                                      mode='min')

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train using input and output data
    model.fit(train_input, train_output,
              validation_split=0.1, callbacks=[early, lr_reduced], epochs=epochs)

    model.summary()
    model.save('Genetic_Visit_AI_model')

    # test the model (validation)
    # write test result and compare (test output) - (ground truth)
    test_prediction = model.predict(test_input)
    test_prediction = np.reshape(test_prediction, (len(test_prediction), 1))
    test_result     = np.concatenate((test_prediction, test_output), axis=1)

    test_result     = pd.DataFrame(test_result)
    test_result.to_csv('newGenetic_train_valid_result.csv')

    # return the trained model
    return model

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)

    input_data  = []
    output_data = []
    times       = 1000

    for i in range(times):
        test(input_data, output_data, i < 10)

    # save dataset
    pd.DataFrame(np.array(input_data)).to_csv('newGeneticDeviceVisitAI_input.csv')
    pd.DataFrame(np.array(output_data)).to_csv('newGeneticDeviceVisitAI_output.csv')

    # load dataset
    inputD  = pd.read_csv('newGeneticDeviceVisitAI_input.csv', index_col=0)
    outputD = pd.read_csv('newGeneticDeviceVisitAI_output.csv', index_col=0)

    # split into training and test data
    train_len    = int(len(inputD) * 0.9)
    
    train_input  = inputD[:train_len]
    train_output = outputD[:train_len]
    test_input   = inputD[train_len:]
    test_output  = outputD[train_len:]

    # training and test
    epochs = 10
    defineModel(train_input, train_output, test_input, test_output, epochs)
