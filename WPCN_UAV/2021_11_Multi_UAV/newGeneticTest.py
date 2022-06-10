import numpy as np
import throughputTest_NewGenetic as T_NG

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)
    
    N = 50

    # device placement (X) :
    #
    #  X . . . X          60
    #          .
    #          .
    #          .
    #          X . . . X  40
    #          .       .
    #          .       .
    #          .       .
    #  X . . . X . . . X  20
    #
    # 10      30      50
    
    deviceList = [[10, 20], [30, 20], [50, 20], [30, 40], [50, 40], [10, 60], [30, 60]]

    # average of x : 30.00 (-> 31.00), y : 37.14, h : 15.00 (as same as the setting)
    initialLocUAV = [31.0, 37.14, 15.0]

    # initial movement
    initialMovement = [3, 1, 2, 5, 4, 6, 0]

    # parameter 1 -> 5 * (random swap probability of two neighboring device)
    param1 = 0.5

    # parameter 2 -> proportion of A and B (with A=0.3, B=0.7)
    param2 = 0.3

    # get optimal path
    (locsUAV, bestMovement, optimalPath) = T_NG.findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, param2)

    print('best movement :')
    print(bestMovement)

    print('optimal path + locations of UAV with the optimal path :')
    for i in range(len(optimalPath)):
        print(i, optimalPath[i], locsUAV[i])
