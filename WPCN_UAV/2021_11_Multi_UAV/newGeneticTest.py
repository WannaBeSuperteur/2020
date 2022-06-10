import numpy as np
import throughputTest_NewGenetic as T_NG

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)
    
    N = 100
    width = 60
    height = 70

    # device placement (X) :
    #
    #  X . . . X          58
    #          .
    #          .
    #          .
    #          X . . . X  39
    #          .       .
    #          .       .
    #          .       .
    #  X . . . X . . . X  20
    #
    # 10      27      44
    
    deviceList = [[10, 20], [27, 20], [44, 20], [27, 39], [44, 39], [10, 58], [27, 58]]

    # average of x : 27.00 (-> 28.00), y : 36.29, h : 15.00 (as same as the setting)
    initialLocUAV = [28.0, 36.29, 15.0]

    # initial movement
    initialMovement = [3, 1, 2, 5, 4, 6, 0]

    # parameter 1 -> 5 * (random swap probability of two neighboring device)
    param1 = 0.5

    # parameter 2 -> proportion of A and B (with A=0.3, B=0.7)
    param2 = 0.3

    # get optimal path
    (locsUAV, bestMovement, optimalPath) = T_NG.findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, param2, width, height)

    print('\ndevice list :')
    print(deviceList)

    print('\nbest movement :')
    print(bestMovement)

    print('\noptimal path + locations of UAV with the optimal path :')
    for i in range(len(optimalPath)):
        print(i, optimalPath[i], locsUAV[i])
