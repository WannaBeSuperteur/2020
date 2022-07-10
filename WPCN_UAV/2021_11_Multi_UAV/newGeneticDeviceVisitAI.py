import throughputTest_NewGenetic as T_NG
import numpy as np
import random
import math

def doBruteForce(deviceList, initialLocUAV, initialMovement):
    pass

def test(input_data, output_data):
    
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
    swappedMovement    = T_NG.swapBasic(deviceList, initialLocUAV, initialMovement, printed)
    bruteForceMovement = doBruteForce(deviceList, initialLocUAV, initialMovement)

    # compute total distance
    totalDistSwapped    = T_NG.computeTotalDist(initialLocUAV, swappedMovement   , deviceList)
    totalDistBruteForce = T_NG.computeTotalDist(initialLocUAV, bruteForceMovement, deviceList)

    # write input and output data
    input_d = []
    
    for i in range(n):
        if i == 0:
            input_data.append(deviceList[0][0])
            input_data.append(deviceList[0][1])
        else:
            input_data.append(deviceList[i][0] - deviceList[i-1][0])
            input_data.append(deviceList[i][1] - deviceList[i-1][1])

    # append to input_data and output_data
    input_data .append(input_d)
    output_data.append([math.log(totalDistBruteForce / totalDistSwapped, 2.0)])

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)

    input_data  = []
    output_data = []
    times       = 1000

    for i in range(times):
        test(input_data, output_data)
