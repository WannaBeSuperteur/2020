import numpy as np
import matplotlib.pyplot as plt
import throughputTest_NewGenetic as T_NG

# save trajectory as graph
def saveTrajectory(width, height, initialLocUAV, locsUAV, deviceList):

    N = len(locsUAV)

    plt.clf()
    plt.suptitle('trajectory result')
    plt.axis([-1, width+1, -1, height+1])

    # for each device
    for i in range(len(deviceList)):
        plt.scatter(deviceList[i][0], deviceList[i][1], s=25, marker='o', c='orange')

    # for each location of UAV
    for t in range(N):
        if t == 0:
            plt.scatter(initialLocUAV[0], initialLocUAV[1], s=25, marker='x', c='black')
            plt.scatter(locsUAV[0][0], locsUAV[0][1], s=25, marker='x', c='forestgreen')
            x = [initialLocUAV[0], locsUAV[0][0]]
            y = [initialLocUAV[1], locsUAV[0][1]]
            
        else:
            plt.scatter(locsUAV[t][0], locsUAV[t][1], s=25, marker='x', c='forestgreen')
            x = [locsUAV[t-1][0], locsUAV[t][0]]
            y = [locsUAV[t-1][1], locsUAV[t][1]]
            
        plt.plot(x, y, linewidth=0.75, c='forestgreen')

    # save the figure
    plt.savefig('newGeneticTest_trajectory.png')

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

    # average of x : 27.00 (-> 20.22), y : 36.29 (-> 20.23), h : 15.00 (as same as the setting)
    initialLocUAV = [20.22, 20.23, 15.0]

    # initial movement
    initialMovement = [3, 1, 2, 5, 4, 6, 0]

    # parameter 1 -> 5 * (random swap probability of two neighboring device)
    param1 = 0.5

    # parameter 2 -> proportion of A and B (with A=0.3, B=0.7)
    param2 = 0.3

    # get optimal path
    (locsUAV, bestMovement, optimalPath) = T_NG.findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, param2, width, height, printed=True)

    print('\ndevice list :')
    print(deviceList)

    print('\nbest movement :')
    print(bestMovement)

    print('\noptimal path + locations of UAV with the optimal path :')
    for i in range(len(optimalPath)):
        print(i, optimalPath[i], locsUAV[i])

    # save trajectory
    saveTrajectory(width, height, initialLocUAV, locsUAV, deviceList)
