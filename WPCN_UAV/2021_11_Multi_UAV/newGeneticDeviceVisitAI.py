import throughputTest_NewGenetic as T_NG
import numpy as np
import random

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

    # test total distance

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)

    input_data  = []
    output_data = []
    times       = 1000

    for i in range(times):
        test(input_data, output_data)
