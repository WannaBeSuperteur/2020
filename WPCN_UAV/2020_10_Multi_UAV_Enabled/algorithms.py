import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def plotClusteringResult(count, L, UAVloc, markerColors, clusters, width, height):
    plt.clf()
    plt.suptitle('clustering: ' + str(count))
            
    for i in range(L):
        plt.scatter(UAVloc[i][0], UAVloc[i][1], s=20, c=markerColors[i])
        for j in range(len(clusters[i])):
            plt.scatter(clusters[i][j][0], clusters[i][j][1], s=3, c=markerColors[i])
                    
    plt.axis([-1, width+1, -1, height+1])

# k Means Clustering algorithm
# return 'UAVs' and 'clusters'

# L          : number of clusters = number of UAVs
# deviceList : list of devices

## UAVs: [UAV0, UAV1, ...]
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is THE POSITION for x, y and h at time t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...

# that is, in the form of UAV[uav_No][x/y/h][time]

## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]
def kMeansClustering(L, deviceList, width, height, H, T, display, saveImg):

    # init clusters
    clusters = [] # clusters to return (each cluster contains corresponding devices)

    # clusters to return (each element indicates corresponding cluster, and contains ID of devices in the cluster)
    # in the form of '0 0 2 1 0 0 1 2 1 2 ' (an example of 3 clusters, 10 devices)
    cluster_mem_now = ''
    cluster_mem_before = ''
    
    UAVloc = [] # location of UAVs ([[x00, y00, h00], [x10, y10, h10], ...])

    for i in range(L):
        clusters.append([])
        UAVloc.append([random.random()*width, random.random()*height, H])

    # marker color for each cluster
    markerColors = []
    for i in range(L):
        mod = ((i * 6) % L) / L
        
        if i < L / 6: markerColors.append('#FF' + format(int(204*mod), '02X') + '00')
        elif i < 2*L / 6: markerColors.append('#' + format(int(255*(1-mod)), '02X') + 'CC00')
        elif i < 3*L / 6: markerColors.append('#00CC' + format(int(255*mod), '02X'))
        elif i < 4*L / 6: markerColors.append('#00' + format(int(204*(1-mod)), '02X') + 'FF')
        elif i < 5*L / 6: markerColors.append('#' + format(int(255*mod), '02X') + '00FF')
        else: markerColors.append('#FF00' + format(int(255*(1-mod)), '02X'))
    
    # do K means clustering
    # cluster = each UAV, centroid = each UAV's location, elements = devices
    count = 0
    while True:
        
        # initialize clusters as all []
        for i in range(L): clusters[i] = []
        cluster_mem_now = ''

        # assign each device to the cluster of the nearest centroid
        for i in range(len(deviceList)):

            thisDevice = deviceList[i] # [Xi, Yi]
            clusterDist = []

            # distance between device i and each cluster
            for j in range(L):
                thisCluster = UAVloc[j] # [xj0, yj0, hj0]

                # distance between device i and cluster j
                dist = math.sqrt(pow(thisDevice[0] - thisCluster[0], 2) + pow(thisDevice[1] - thisCluster[1], 2))
                clusterDist.append(dist)

            # find the cluster with the minimum distance
            minDistCluster = 0 # index of the cluster with minimum distance
            minDist = clusterDist[0]
            
            for j in range(L):
                if clusterDist[j] < minDist:
                    minDist = clusterDist[j]
                    minDistCluster = j

            # append to the cluster
            clusters[minDistCluster].append(thisDevice)
            cluster_mem_now += str(minDistCluster) + ' '

        # update each centroid as the mean of the cluster members
        for i in range(L):

            # mean of the cluster members (devices)
            mean = [0, 0]

            # for each device in the cluster
            thisCluster = clusters[i]

            # do not compute if there is no device in the cluster
            if len(thisCluster) == 0: continue

            # compute the mean
            for j in range(len(thisCluster)):
                thisDevice = thisCluster[j]
                mean[0] += thisDevice[0]
                mean[1] += thisDevice[1]

            # update the centroid ( = UAVloc )
            mean[0] /= len(thisCluster)
            mean[1] /= len(thisCluster)

            UAVloc[i][0] = mean[0]
            UAVloc[i][1] = mean[1]

        # print or save info
        if display == True or saveImg == True:
            
            # print cluster info
            if display == True:
                plotClusteringResult(count, L, UAVloc, markerColors, clusters, width, height)
                plt.show()

            # save image
            if saveImg == True:
                plotClusteringResult(count, L, UAVloc, markerColors, clusters, width, height)
                plt.savefig('clustering_result.png')

        # break for the same cluster
        print('\nclustering ' + str(count))
        print('before : ' + cluster_mem_before)
        print('after  : ' + cluster_mem_now)
        if cluster_mem_now == cluster_mem_before: break

        # update cluster_mem_before
        cluster_mem_before = cluster_mem_now

        # increase count
        count += 1

    # return UAVs and clusters (not moving, stay at the initial position)
    # each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...

    UAVs = []
    for i in range(len(UAVloc)):
        thisUAV = []

        # append x, y and h to the UAV
        for j in range(3): # for each (x, y or h)
            temp = []
            for t in range(T + 1): temp.append(UAVloc[i][j])
            thisUAV.append(temp)
            
        # append to list of UAVs
        UAVs.append(thisUAV)

    # return
    return (UAVs, clusters, cluster_mem_now)

if __name__ == '__main__':

    # number of clusters = number of UAVs
    L = 3

    # number of devices
    deviceList = []
    devices = 50
    width = 30
    height = 30
    H = 15
    T = 10
    
    # randomly place devices
    for i in range(devices):
        xVal = random.random() * width
        yVal = random.random() * height
        device_i = [xVal, yVal]
        deviceList.append(device_i)

    # do K means clustering
    (UAVs, clusters, clusterNOs) = kMeansClustering(L, deviceList, width, height, H, T, True, True)

    # print result
    print(' << UAV location >>\n')
    for i in range(L): print(UAVs[i][0])
    print('\n << clusters >>')
    for i in range(L):
        print('\ncluster No. ' + str(i))
        for j in range(len(clusters[i])):
            print(clusters[i][j])
