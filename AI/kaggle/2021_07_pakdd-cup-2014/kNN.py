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
def aggregateRepairCount():

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
def distance_MX(val0, val1, MX_corrcoef):

    M_index0 = int(val0[1:]) - 1
    M_index1 = int(val1[1:]) - 1
        
    return 1.0 - MX_corrcoef[M_index0][M_index1]

# PX : P01 ~ P31
def distance_PX(val0, val1, PX_corrcoef):

    P_index0 = int(val0[1:]) - 1
    P_index1 = int(val1[1:]) - 1
        
    return 1.0 - PX_corrcoef[P_index0][P_index1]

# month : 1 ~ 12, using weight*sin(DIF/2) for angle DIF
# weight = 0.3
def distance_month(val0, val1):

    weight = 0.3

    angle_val0 = float(val0) * math.pi / 6.0
    angle_val1 = float(val1) * math.pi / 6.0
    DIF = abs(angle_val0 - angle_val1)

    return weight * math.sin(DIF/2)

# year*12 + month : abs(distance)/24
def distance_ym(val0, val1):
    distance = abs(int(val0) - int(val1))
    return distance / 24

# kNN for [MX, PX, year(repair), year*12+month(repair)] and make final submission
def kNN(aggregated, MX_corrcoef, PX_corrcoef, valid, N, limits):

    N_float = N % 1.0
    N = math.ceil(N)
    
    # read mapping data
    # train -> Output_TargetID_Mapping.csv
    # valid -> aggregate_repair.csv
    
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

        repairYM = repairYM_train

    else: # test
        mapping = pd.read_csv('Output_TargetID_Mapping.csv')
        mapping = np.array(mapping)
        
        repairYM = repairYM_train + repairYM_valid

    print('\n === valid ===')
    print(valid)
    print('\n === aggregated ===')
    print(aggregated)
    print('\n === mapping ===')
    print(mapping)

    prediction = []
    explanation = []

    if valid == True:
        state_text = 'valid'
    else:
        state_text = 'test'

    for i in range(len(mapping)):
        print(i)

        map_i = mapping[i]
        
        val0_M = map_i[0]
        val0_P = map_i[1]
        val0_mo = map_i[3]
        val0_ym = int(map_i[2]) * 12 + int(val0_mo)

        # compute distance
        dist_aggregated = []

        for j in range(Ms * Ps * repairYM):

            agg_j = aggregated[j]

            val1_M = agg_j[0]
            val1_P = agg_j[1]
            val1_mo = agg_j[3]
            val1_ym = int(agg_j[2]) * 12 + int(val1_mo)

            # find distance between the set of val0 and val1
            dist_M  = distance_MX   (val0_M , val1_M , MX_corrcoef)
            dist_P  = distance_PX   (val0_P , val1_P , PX_corrcoef)
            dist_mo = distance_month(val0_mo, val1_mo)
            dist_ym = distance_ym   (val0_ym, val1_ym)
            
            dist_total = math.sqrt(dist_M*dist_M + dist_P*dist_P + dist_mo*dist_mo + dist_ym*dist_ym)

            # save [index, distance, repair count, val1_M, val1_P, val1_year, val1_month]
            dist_aggregated.append([j, dist_total, float(agg_j[4]),
                                    val1_M, val1_P, agg_j[2], val1_mo])

        # sort distance
        dist_aggregated.sort(key=lambda x:x[1])
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

            # write explanation
            # [val0_M, val0_P, val0_year, val0_month, rank,
            #  index, distance, repair count, val1_M, val1_P, val1_year, val1_month]
            explanation.append([val0_M, val0_P, map_i[2], val0_mo, j,
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

    return MAEs

# 0. aggregate repair count : group by (MX, PX, year(repair), month(repair))

# 1. compute correlation coefficient CORR for each MX, using repair count for each year(repair), month(repair) and PX
#    compute correlation coefficient CORR for each PX, using repair count for each year(repair), month(repair) and MX

# 2. using kNN by module type(MX), component type(PX), month(repair) and year*12+month(repair)
#    using (1 - CORR)       for different MX and PX,
#          0~weight         for month,
#          abs(distance)/24 for year*12+month(repair)
if __name__ == '__main__':
    
    repairAggregated = aggregateRepairCount()
    pd.DataFrame(repairAggregated).to_csv('kNN_aggregate.csv')
    
    MX_corrcoef = computeCorrCoef_MX(repairAggregated)
    pd.DataFrame(MX_corrcoef).to_csv('kNN_MX_corr.csv')
    
    MP_corrcoef = computeCorrCoef_PX(repairAggregated)
    pd.DataFrame(MP_corrcoef).to_csv('kNN_MP_corr.csv')

    valid = False
    test_N = 1 # best valid N
    test_limit = 348 # 1700 (best valid limit) * 6.859 (test average) / 33.49283 (valid average)

    Ns = [1, 1.1, 1.25, 1.5, 2, 3, 4, 5, 6, 7, 10, 12, 15, 17, 20, 25, 30, 40, 50, 75, 100]

    # apply test limits PROPORTIONALLY to the average of all repair_count
    limits = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0,
              5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
              60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 130.0, 150.0, 175.0, 200.0,
              225.0, 250.0, 275.0, 300.0, 330.0, 360.0, 400.0, 450.0, 500.0, 550.0,
              600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1100.0,
              1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2130.0,
              2260.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0]

    # validation
    if valid == True:
        MAE_array = []
        MAE_array.append(limits)
        count = 0
        
        for N in Ns:
            count += 1
            
            MAEs = kNN(repairAggregated, MX_corrcoef, MP_corrcoef, True, N, limits)
            MAE_array.append(MAEs)

            MAE_df = np.array(MAE_array)
            MAE_df = MAE_df.T
            MAE_df = pd.DataFrame(MAE_df)
            
            MAE_df.columns = np.array([-1] + Ns[:count]).astype(str)
            MAE_df.to_csv('kNN_valid_MAE.csv')

    # test
    else:
        kNN(repairAggregated, MX_corrcoef, MP_corrcoef, False, test_N, [test_limit])