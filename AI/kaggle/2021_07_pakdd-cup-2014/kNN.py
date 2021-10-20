import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error

# to do : find the best result for each # of neighbors (N) and the limit

Ms = 9 # M1~M9
Ps = 31 # P01~P31
repairYM_train = 47 # repair year and month, 2005/02 ~ 2009/03 for training
repairYM_valid = 12 # repair year and month, 2009/04 ~ 2009/12 for validation

# sum of repairYM_train and repairYM_valid MUST be 59

# aggregate repair count from RepairTrain.csv
def aggregateRepairCount(valid):

    if valid == True:
        repairYM = repairYM_train
    else:
        repairYM = repairYM_train + repairYM_valid

    aggregated = np.zeros((Ms * Ps * repairYM, 5)).astype(str)

    # aggregation columns
    for i in range(Ms * Ps * repairYM):
        aggregated[i][0] = 'M' + str(i // (Ps * repairYM) + 1)
        aggregated[i][1] = 'P' + '%02d' % (i // repairYM % Ps + 1)
        aggregated[i][2] = str((i % repairYM + 2004 * 12 + 1) // 12 + 1)
        aggregated[i][3] = (i % repairYM + 1) % 12 + 1

    # read RepairTrain.csv and add values and fill aggregated array
    repairTrain = pd.read_csv('RepairTrain.csv')
    repairTrain = np.array(repairTrain)

    for i in range(len(repairTrain)):
        M = int(repairTrain[i][0][1:])
        P = int(repairTrain[i][1][1:])
        year = int(repairTrain[i][3].split('/')[0])
        month = int(repairTrain[i][3].split('/')[1])
        repair_count = int(repairTrain[i][4])

        MP_index = (M - 1) * (Ps * repairYM) + (P - 1) * repairYM
        ym_index = (year - 2005) * 12 + (month - 2)
        aggregated_index = MP_index + ym_index

        if ym_index >= repairYM: continue

        aggregated[aggregated_index][4] = float(aggregated[aggregated_index][4]) + repair_count

    return aggregated

# compure correlation coefficient for each MX
# using repair count for each year(repair), month(repair) and PX
def computeCorrCoef_MX(aggregated):

    repairYM = repairYM_train + repairYM_valid

    # repair count for each MX (M1~M9)
    MX_arrays = []
    for i in range(Ms):
        MX_arrays.append([])

    for i in range(Ms * Ps * repairYM):
        M_no = int(aggregated[i][0][1:])
        MX_arrays[M_no - 1].append(float(aggregated[i][4]))

    # convert into numpy array
    MX_arrays = np.array(MX_arrays).astype(float)

    # compute correlation coefficient
    MX_corrcoef = np.corrcoef(MX_arrays)
    return MX_corrcoef

# compure correlation coefficient for each PX
# using repair count for each year(repair), month(repair) and MX
def computeCorrCoef_PX(aggregated):

    repairYM = repairYM_train + repairYM_valid

    # repair count for each PX (P01~P31)
    PX_arrays = []
    for i in range(Ps):
        PX_arrays.append([])

    for i in range(Ms * Ps * repairYM):
        P_no = int(aggregated[i][1][1:])
        PX_arrays[P_no - 1].append(float(aggregated[i][4]))

    # convert into numpy array
    PX_arrays = np.array(PX_arrays).astype(float)

    # compute correlation coefficient
    PX_corrcoef = np.corrcoef(PX_arrays)
    return PX_corrcoef

# compute distance for each value
# using (1 - CORR)       for different MX and PX,
#       0~2              for month,
#       abs(distance)/24 for year*12+month(repair)

# MX : M1 ~ M9
def distance_MX(val0, val1, MX_corrcoef, weight):

    M_index0 = int(val0[1:]) - 1
    M_index1 = int(val1[1:]) - 1
        
    return weight * (1.0 - MX_corrcoef[M_index0][M_index1])

# PX : P01 ~ P31
def distance_PX(val0, val1, PX_corrcoef, weight):

    P_index0 = int(val0[1:]) - 1
    P_index1 = int(val1[1:]) - 1
        
    return weight * (1.0 - PX_corrcoef[P_index0][P_index1])

# month : 1 ~ 12, using weight*sin(DIF/2) for angle DIF
def distance_month(val0, val1, weight):
    angle_val0 = float(val0) * math.pi / 6.0
    angle_val1 = float(val1) * math.pi / 6.0
    DIF = abs(angle_val0 - angle_val1)

    return weight * 0.25 * math.sin(DIF/2)

# year*12 + month : abs(distance)/24
def distance_ym(val0, val1, weight):
    distance = abs(int(val0) - int(val1))
    return weight * distance / 24

# get distance
def getDistanceArray(mapping, aggregated, repairYM, MX_corrcoef, PX_corrcoef):

    print('getting distance array ...')
    distTotal = np.zeros((len(mapping), Ms * Ps * repairYM, 4))

    for i in range(len(mapping)):
        
        if i % 15 == 0:
            print(str(i) + ' / ' + str(len(mapping)))

        map_i = mapping[i]
        
        val0_M = map_i[0]
        val0_P = map_i[1]
        val0_mo = map_i[3]
        val0_ym = int(map_i[2]) * 12 + int(val0_mo)

        # compute distance
        dist_aggregated = [None for i in range(2)]
        
        for j in range(Ms * Ps * repairYM):

            agg_j = aggregated[j]

            val1_M = agg_j[0]
            val1_P = agg_j[1]
            val1_mo = agg_j[3]
            val1_ym = int(agg_j[2]) * 12 + int(val1_mo)

            # find distance between the set of val0 and val1 (with base weight 1.0)
            dist_M  = distance_MX   (val0_M , val1_M , MX_corrcoef, 1.0)
            dist_P  = distance_PX   (val0_P , val1_P , PX_corrcoef, 1.0)
            dist_mo = distance_month(val0_mo, val1_mo, 1.0)
            dist_ym = distance_ym   (val0_ym, val1_ym, 1.0)
            
            distTotal[i][j] = np.array([dist_M, dist_P, dist_mo, dist_ym])

    return distTotal

# mapping for training and validation
def getMapping(aggregated, valid):

    if valid == True: # valid
        mapping = pd.read_csv('kNN_aggregate.csv')
        mapping = np.array(mapping)

        # remove ID column
        mapping = np.delete(mapping, 0, 1)

        # remove rows with training months from mapping
        for i in range(len(mapping)-1, -1, -1):
            ym = (mapping[i][2] - 2005) * 12 + (mapping[i][3] - 2)
                
            if ym < repairYM_train:
                mapping = np.delete(mapping, i, 0)

        # save mapping array
        pd.DataFrame(mapping).to_csv('kNN_valid_groundTruth.csv')

        # remove repair count column
        mapping = np.delete(mapping, 4, 1)

        # remove validation data of aggregated array
        for i in range(len(aggregated)-1, -1, -1):
            ym = (int(aggregated[i][2]) - 2005) * 12 + (int(aggregated[i][3]) - 2)
                
            if ym >= repairYM_train:
                aggregated = np.delete(aggregated, i, 0)

    else: # test
        mapping = pd.read_csv('Output_TargetID_Mapping.csv')
        mapping = np.array(mapping)

    return mapping
                
# kNN for [MX, PX, year(repair), year*12+month(repair)] and make final submission
def kNN(aggregated, mapping, MX_corrcoef, PX_corrcoef, valid, N, limits,
        dMX_weight, dPX_weight, dm_weight, dYM_weight, distTotal):

    N_float = N % 1.0
    N = math.ceil(N)

    print('shape of mapping:')
    print(np.shape(mapping))
    print('shape of aggregated:')
    print(np.shape(aggregated))

    print('saving...')
    if valid == True:
        pd.DataFrame(mapping).to_csv('valid_getDistanceArray_mapping.csv')
        pd.DataFrame(aggregated).to_csv('valid_getDistanceArray_aggregated.csv')
    else:
        pd.DataFrame(mapping).to_csv('test_getDistanceArray_mapping.csv')
        pd.DataFrame(aggregated).to_csv('test_getDistanceArray_aggregated.csv')
    
    # read mapping data
    # train -> Output_TargetID_Mapping.csv
    # valid -> aggregate_repair.csv
    
    if valid == True: # valid
        repairYM = repairYM_train
    else: # test
        repairYM = repairYM_train + repairYM_valid

    print('\n === N:' + str(N) + ', weights:' + str([dMX_weight, dPX_weight, dm_weight, dYM_weight]) + ' ===')

    prediction = []
    explanation = []

    if valid == True:
        state_text = 'valid'
    else:
        state_text = 'test'
    
    # running
    for i in range(len(mapping)):
        
        if i % 100 == 0:
            print(str(i) + ' / ' + str(len(mapping)))

        # compute distance
        dist_aggregated = [None for i in range(2)]
        
        for j in range(Ms * Ps * repairYM):

            # find distance
            dist_0 = distTotal[i][j][0] * dMX_weight
            dist_1 = distTotal[i][j][1] * dPX_weight
            dist_2 = distTotal[i][j][2] * dm_weight
            dist_3 = distTotal[i][j][3] * dYM_weight

            # pass this cell (with distance 0)
            if dist_0 == 0 and dist_1 == 0 and dist_2 == 0 and dist_3 == 0: continue

            dist_total = dist_0 * dist_0 + dist_1 * dist_1 + dist_2 * dist_2 + dist_3 * dist_3

            # val1_M, val1_P, val1_year and val1_month
            val1_M = aggregated[j][0]
            val1_P = aggregated[j][1]
            val1_yr = aggregated[j][2]
            val1_mo = aggregated[j][3]
            
            # save [index, distance, repair count, val1_M, val1_P, val1_year, val1_month]
            # index 0 is None
            if dist_aggregated[0] == None:
                dist_aggregated[0] = [j, dist_total, float(aggregated[j][4]), val1_M, val1_P, val1_yr, val1_mo]

            # (newDist) < dist[0]
            elif dist_total < dist_aggregated[0][1]:
                dist_aggregated[1] = dist_aggregated[0]
                dist_aggregated[0] = [j, dist_total, float(aggregated[j][4]), val1_M, val1_P, val1_yr, val1_mo]

            # (newDist) >= dist[0] and index 1 is None
            elif dist_total >= dist_aggregated[0][1] and dist_aggregated[1] == None:
                dist_aggregated[1] = [j, dist_total, float(aggregated[j][4]), val1_M, val1_P, val1_yr, val1_mo]

            # (newDist) >= dist[0] and (newDist) < dist[1]
            elif dist_total >= dist_aggregated[0][1] and dist_total < dist_aggregated[1][1]:
                dist_aggregated[1] = [j, dist_total, float(aggregated[j][4]), val1_M, val1_P, val1_yr, val1_mo]

        # (do not need to sort distance)
        dist_aggregated = np.array(dist_aggregated)

        # weighted sum = repairSum / distanceSum
        #              = sum of (repair_count * 1 / distance) / sum of (1 / distance)
        repairSum = 0.0
        distanceSum = 0.0
        
        for j in range(N):

            if j == N-1: # last part
                if N_float == 0: # (last) integer part
                    weight = 1.0
                    
                else: # (last) float part
                    weight = N_float
                
            else: # (not last) integer part
                weight = 1.0

            repairSum += weight * float(dist_aggregated[j][2]) / float(dist_aggregated[j][1])
            distanceSum += weight / float(dist_aggregated[j][1])

            # val0_M, val0_P, val0_year and val0_month
            val0_M = mapping[i][0]
            val0_P = mapping[i][1]
            val0_yr = mapping[i][2]
            val0_mo = mapping[i][3]

            # write explanation
            # [val0_M, val0_P, val0_year, val0_month, rank,
            #  index, distance, repair count, val1_M, val1_P, val1_year, val1_month]
            explanation.append([val0_M, val0_P, val0_yr, val0_mo, j,
                                dist_aggregated[j][0], dist_aggregated[j][1],
                                dist_aggregated[j][2], dist_aggregated[j][3],
                                dist_aggregated[j][4], dist_aggregated[j][5],
                                dist_aggregated[j][6]])

        prediction.append(repairSum / distanceSum)

        # write prediction periodically
        if (i + 1) % 500 == 0:
            print('writing prediction ...')
            pd.DataFrame(prediction).to_csv('kNN_' + state_text + '_prediction_' + str(i) + '.csv')
            pd.DataFrame(explanation).to_csv('kNN_' + state_text + '_explanation_' + str(i) + '.csv')

    # for each limit value,
    MAEs = []
    minMAE = 1000000.0
    minMAE_limit = -1
    
    for limit in limits:
        
        # write valid prediction and compute error (if valid)
        if valid == True:
            valid_groundTruth = pd.read_csv('kNN_valid_groundTruth.csv', index_col=0)
            valid_groundTruth = np.array(valid_groundTruth)

            # compare prediction with valid_groundTruth and compute mean absolute error
            valid_groundTruth_values = valid_groundTruth[:, 4]
            prediction_values = np.array(prediction)
            prediction_values = np.clip(prediction_values, 0, limit)

            MAE = mean_absolute_error(valid_groundTruth_values, prediction_values)
            print('MAE with N=' + ('%4d' % N) + ', limit=' + ('%2.4f' % limit) + ' : ' + str(round(MAE, 6)))
            
            MAEs.append(MAE)

            # update minMAE and the index of it
            if MAE < minMAE:
                minMAE = MAE
                minMAE_limit = limit

        # write final prediction (if test)
        else:
            prediction = pd.DataFrame(prediction)
            prediction_with_limit = prediction.clip(0, limit)
            explanation = pd.DataFrame(explanation)

            # change row index (0..N-1 to 1..N) for final prediction
            prediction_with_limit = prediction_with_limit.rename(index=lambda x:x+1)
            
            prediction.to_csv('kNN_test_prediction.csv')
            prediction_with_limit.to_csv('kNN_test_prediction_limit_' + str(limit) + '.csv')
            explanation.to_csv('kNN_test_explanation.csv')

    return [MAEs, minMAE, minMAE_limit]

# 0. aggregate repair count : group by (MX, PX, year(repair), month(repair))

# 1. compute correlation coefficient CORR for each MX, using repair count for each year(repair), month(repair) and PX
#    compute correlation coefficient CORR for each PX, using repair count for each year(repair), month(repair) and MX

# 2. using kNN by module type(MX), component type(PX), month(repair) and year*12+month(repair)
#    using (1 - CORR)       for different MX and PX,
#          0~weight         for month,
#          abs(distance)/24 for year*12+month(repair)
def main(valid):
    
    repairAggregated = aggregateRepairCount(False)
    repairAggregated_valid = aggregateRepairCount(True)
    
    pd.DataFrame(repairAggregated).to_csv('kNN_aggregate.csv')
    pd.DataFrame(repairAggregated_valid).to_csv('kNN_aggregate_valid.csv')
    
    MX_corrcoef = computeCorrCoef_MX(repairAggregated)
    pd.DataFrame(MX_corrcoef).to_csv('kNN_MX_corr.csv')
    
    PX_corrcoef = computeCorrCoef_PX(repairAggregated)
    pd.DataFrame(PX_corrcoef).to_csv('kNN_PX_corr.csv')

    test_N = 1 # best valid N
    test_limit = 70 # 348 = 1700 (best valid limit) * 6.859 (test average) / 33.49283 (valid average)

    # weights for test: [dMX_weight, dPX_weight, dm_weight, dYM_weight]
    test_weights = [0.25, 0.25, 0.25, 0.25]

    # number of neighbors
    N = 1

    # limits
    # apply test limits PROPORTIONALLY to the average of all repair_count
    limits = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0,
              5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
              60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 130.0, 150.0, 175.0, 200.0,
              225.0, 250.0, 275.0, 300.0, 330.0, 360.0, 400.0, 450.0, 500.0, 550.0,
              600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1100.0,
              1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2125.0,
              2250.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0]

    # initial weights: [dMX_weight, dPX_weight, dm_weight, dYM_weight]
    weights = [0.25, 0.25, 0.25, 0.25]
    wcr = 0.01
    round_ = 0

    # get distance array first
    # shape: (len(mapping), len(aggregated), 4) = (len(mapping), Ms * Ps * repairYM, 4)
    validMapping = getMapping(repairAggregated_valid, True)
    testMapping = getMapping(repairAggregated, False)
    
    distTotal_valid = getDistanceArray(validMapping, repairAggregated_valid, repairYM_train,
                                       MX_corrcoef, PX_corrcoef)
    
    distTotal_test = getDistanceArray(testMapping, repairAggregated, repairYM_train + repairYM_valid,
                                      MX_corrcoef, PX_corrcoef)

    # validation
    if valid == True:

        # valid result array
        valid_result = []

        # use hill-climbing search
        while True:
            print('round:', round_)

            # get minimum MAE
            [MAEs, minMAE, minMAE_limit] = kNN(repairAggregated_valid, validMapping,
                                               MX_corrcoef, PX_corrcoef, True, N, limits,
                                               weights[0], weights[1], weights[2], weights[3],
                                               distTotal_valid)

            valid_result.append([MAEs, minMAE, minMAE_limit, weights[0], weights[1], weights[2], weights[3]])

            # find neighbors (with 12 options)
            neighbor_options = [[wcr, -wcr, 0, 0], [wcr, 0, -wcr, 0], [wcr, 0, 0, -wcr],
                                [-wcr, wcr, 0, 0], [0, wcr, -wcr, 0], [0, wcr, 0, -wcr],
                                [-wcr, 0, wcr, 0], [0, -wcr, wcr, 0], [0, 0, wcr, -wcr],
                                [-wcr, 0, 0, wcr], [0, -wcr, 0, wcr], [0, 0, -wcr, wcr]]

            minMAE_neighbors = []

            # get minimum MAE of each neighbor
            for i in range(12):
                nei = list(np.array(weights) + np.array(neighbor_options[i]))

                [_, minMAE_nei, _] = kNN(repairAggregated_valid, validMapping,
                                         MX_corrcoef, PX_corrcoef, True, N, limits,
                                         nei[0], nei[1], nei[2], nei[3],
                                         distTotal_valid)

                minMAE_neighbors.append(minMAE_nei)

            # compare the minMAE
            goto = -1
            minMAE_updated = minMAE
            
            for i in range(12):
                print('round:', round_, ' neighbor:', i)
                
                if minMAE_neighbors[i] < minMAE_updated:
                    minMAE_updated = minMAE_neighbors[i]
                    goto = i

            # go to the neighbor
            if goto >= 0:
                weight = [weight[0] + neighbor_options[goto][0],
                          weight[1] + neighbor_options[goto][1],
                          weight[2] + neighbor_options[goto][2],
                          weight[3] + neighbor_options[goto][3]]
            else:
                break

        # save validation result
        valid_result = pd.DataFrame(np.array(valid_result))
        valid_result.to_csv('valid_result.csv')

    # test
    else:
        kNN(repairAggregated, testMapping,
            MX_corrcoef, PX_corrcoef, False, test_N, [test_limit],
            test_weights[0], test_weights[1], test_weights[2], test_weights[3],
            distTotal_test)

if __name__ == '__main__':
    main(True) # valid mode
    main(False) # test mode
