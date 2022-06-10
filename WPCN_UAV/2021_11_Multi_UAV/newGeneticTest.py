import throughputTest_NewGenetic as T_NG

if __name__ == '__main__':
    
    N = 30

    # device placement (X) :
    #
    #  X . . . X          28
    #          .
    #          .
    #          .
    #          X . . . X  24
    #          .       .
    #          .       .
    #          .       .
    #  X . . . X . . . X  20
    #
    # 10      14      18
    
    deviceList = [[10, 20], [14, 20], [18, 20], [14, 24], [18, 24], [10, 28], [14, 28]]

    # average of x : 14.00, y : 23.43, h : 15.00 (as same as the setting)
    initialLocUAV = [14.0, 23.43, 15.0]

    # initial movement
    initialMovement = [3, 1, 2, 5, 4, 6, 0]

    # parameter 1 -> 5 * (random swap probability of two neighboring device)
    param1 = 0.5

    # parameter 2 -> proportion of A and B (with A=0.3, B=0.7)
    param2 = 0.3

    # get optimal path
    (bestMovement, optimalPath) = T_NG.findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, param2)

    print('best movement :')
    print(bestMovement)

    print('optimal path :')
    print(optimalPath)
