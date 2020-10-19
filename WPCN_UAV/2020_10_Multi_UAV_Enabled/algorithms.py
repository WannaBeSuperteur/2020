import math

# k Means Clustering algorithm
# return 'UAVs' and 'clusters'

# L          : number of clusters = number of UAVs
# deviceList : list of devices

## UAVs: [UAV0, UAV1, ...]
# each UAV : UAV0 = [x0, y0, h0], UAV1 = [x1, y1, h1], ...
# each element of x0, y0 and h0 is the vector of location, from previous t to current t
# where x0 = [x00, x01, ..., x0t], x1 = [x10, x11, ..., x1t], ...
#       y0 = [y00, y01, ..., y0t], ...
#       h0 = [h00, h01, ..., h0t], ...

## clusters: [c0_deviceList, c1_deviceList, ...]
# cK_deviceList: device list of cluster k,
#                in the form of [dev0, dev1, ...] == [[X0, Y0], [X1, Y1], ...]
def kMeansClustering(L, deviceList):
    
