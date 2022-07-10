import numpy as np
import matplotlib.pyplot as plt
import throughputTest_NewGenetic as T_NG
from matplotlib.pyplot import figure

# save trajectory as graph
def saveTrajectory(width, height, initialLocUAV, locsUAV, deviceList):

    N = len(locsUAV)

    plt.clf()
    plt.figure(figsize=(width / 6.0, height / 6.0))
    plt.suptitle('trajectory result')
    plt.axis([-1, width+1, -1, height+1])

    # for each device
    for i in range(len(deviceList)):
        plt.scatter(deviceList[i][0], deviceList[i][1], s=70, marker='o', c='orange')

    # for each location of UAV
    doNotMoveCnt = 0
    
    for t in range(N):
        if t == 0:
            plt.scatter(initialLocUAV[0], initialLocUAV[1], s=100, marker='X', c='black')
            plt.scatter(locsUAV[0][0], locsUAV[0][1], s=35, marker='x', c='forestgreen')
            
            x = [initialLocUAV[0], initialLocUAV[0] * 0.25 + locsUAV[0][0] * 0.75, locsUAV[0][0]]
            y = [initialLocUAV[1], initialLocUAV[1] * 0.25 + locsUAV[0][1] * 0.75, locsUAV[0][1]]
            
        else:
            if t == N-1:
                plt.scatter(locsUAV[t][0], locsUAV[t][1], s=100, marker='X', c='blue')
            else:
                plt.scatter(locsUAV[t][0], locsUAV[t][1], s=35, marker='x', c='forestgreen')
                
            x = [locsUAV[t-1][0], locsUAV[t-1][0] * 0.25 + locsUAV[t][0] * 0.75, locsUAV[t][0]]
            y = [locsUAV[t-1][1], locsUAV[t-1][1] * 0.25 + locsUAV[t][1] * 0.75, locsUAV[t][1]]

        # check stop of UAV
        if pow(x[1] - x[0], 2) + pow(y[1] - y[0], 2) < 0.1 and t < N-2:
            doNotMoveCnt += 1
        else:
            # write "stop count" when starting moving or the last time slot
            if doNotMoveCnt > 0:
                plt.text(x[0], y[0], s=str(doNotMoveCnt + 1), fontsize=10)
            doNotMoveCnt = 0

        plt.plot(x[:2], y[:2], linewidth=0.75, c='forestgreen')
        plt.plot(x[1:], y[1:], linewidth=2.5 , c='forestgreen')

    # save the figure
    plt.savefig('newGeneticTest_trajectory.png')

if __name__ == '__main__':

    # numpy setting
    np.set_printoptions(edgeitems=20, linewidth=200)
    
    N = 300
    width = 60
    height = 70

    deviceList = [[5, 5], [20, 5], [35, 5], [55, 5],
                  [5, 25], [20, 25], [35, 25], [55, 25],
                  [5, 45], [20, 45], [35, 45], [55, 45],
                  [5, 65], [20, 65], [35, 65], [55, 65]]

    # average of x : 27.00 (-> 20.22), y : 36.29 (-> 20.23), h : 15.00 (as same as the setting)
    initialLocUAV = [20.22, 20.23, 15.0]

    # initial movement
    initialMovement = [7, 1, 10, 2, 9, 4, 0, 5, 6, 13, 3, 14, 8, 12, 15, 11]

    # parameter 1 -> the value of x, where the stop times of UAV is decided by (distance between UAV and nearest device)^x
    param1 = 0.4

    # get optimal path
    (locsUAV, bestMovement, optimalPath) = T_NG.findOptimalPath(N, deviceList, initialLocUAV, initialMovement, param1, width, height, printed=True)

    print('\ndevice list :')
    print(deviceList)

    print('\nbest movement :')
    print(bestMovement)

    print('\noptimal path + locations of UAV with the optimal path :')
    for i in range(len(optimalPath)):
        print(i, optimalPath[i], locsUAV[i])

    # save trajectory
    saveTrajectory(width, height, initialLocUAV, locsUAV, deviceList)
