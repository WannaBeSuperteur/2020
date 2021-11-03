import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import formula as f

def plotClusteringResult(count, L, UAVloc, markerColors, clusters, width, height):
    plt.clf()
    plt.suptitle('clustering: ' + str(count))
            
    for i in range(L):
        plt.scatter(UAVloc[i][0], UAVloc[i][1], s=20, c=markerColors[i])
        for j in range(len(clusters[i])):
            plt.scatter(clusters[i][j][0], clusters[i][j][1], s=3, c=markerColors[i])
                    
    plt.axis([-1, width+1, -1, height+1])

### k Means Clustering algorithm
# L          : number of clusters = number of UAVs
# deviceList : list of devices
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

    # wkl, ql(t) : wkl = (xkl, ykl, 0), ql(t) = (xl(t), yl(t), hl(t)), h_min <= hl(t) <= h_max
    #              where w = [[l, k, xkl, ykl, 0], ...]
    #                and q = [[l, t, xlt, ylt, hlt], ...]

    # write ql(t): UAVs info (initially not moving, stay at the initial position first)
    # in the form of q = [[l, t, xlt, ylt, hlt], ...]

    q = []
    for l in range(L):
        for t in range(T+1):
            q.append([l, t, UAVloc[l][0], UAVloc[l][1], UAVloc[l][2]])

    # write wkl: device location info (not moving)
    # in the form of w = [[l, k, xkl, ykl, 0], ...]
    w = []
    for l in range(L):
        for k in range(len(clusters[l])):
            w.append([l, k, clusters[l][k][0], clusters[l][k][1], 0])

    # return
    return (q, w)

# decide whether to transfer energy for each UAV

# al[0]     : denote downlink WET node of UAV l
#             equal 0 or 1, energy is transferred or not by the l-th UAV
# assumption : ALWAYS transfer energy to a device
def transferEnergy(cluster, l):
    return 1

# find the device to communicate with

# a_l,kl[n] : uplink WIT allocation between UAV l and IoT device kl at n-th time slot
#             equal 0 or 1, IoT device kl does communicate/or not with l-th UAV
# alkl      : a_l,kl[n] for each UAV l, device k and time slot n
#             where alkl = [[l0, k, l1, n, value], ...]

# (device to communicate) = (the device with best condition, with the largest value of g[n][l][k_l])
# within the GIVEN cluster
# w = [[l, k, xkl, ykl, 0], ...]
# q = [[l, t, xlt, ylt, hlt], ...]
def findDeviceToCommunicate(q, w, l, n, T, s, b1, b2, mu1, mu2, fc, c, alpha):

    # find the start index of w[x] = [l, 0, ...]
    start_index = f.find_wkl(w, 0, l)
    end_index   = f.find_wkl(w, 0, l+1)
    
    devicesInThisCluster = end_index - start_index
    condition = [0 for i in range(devicesInThisCluster)]

    # compute g[n][l][k_l] for each device in the cluster
    for k in range(start_index, end_index):
        g_value = f.formula_04(q, w, k, l, l, n, T, s, b1, b2, mu1, mu2, fc, c, alpha)
        condition.append(g_value)
    
    deviceToCommunicate = np.argmax(condition)

    # return the index of the device to communicate
    return deviceToCommunicate

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
    (q, w) = kMeansClustering(L, deviceList, width, height, H, T, True, True)

    # print result
    print(' << q >>')
    print(np.array(q))
    
    print('\n << w >>')
    print(np.array(w))
