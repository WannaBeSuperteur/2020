import numpy as np
import pandas as pd
import math

Ms = 9 # M1~M9
Ps = 31 # P01~P31
repairYM = 59 # repair year and month, 2005/02 ~ 2009/12

# aggregate repair count from RepairTrain.csv
def aggregateRepairCount():
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

    pd.DataFrame(aggregated).to_csv('aggregate_repair.csv')
    return aggregated

# compure correlation coefficient for each MX
# using repair count for each year(repair), month(repair) and PX
def computeCorrCoef_MX(aggregated):

    # repair count for each MX (M1~M9)
    MX_arrays = []
    for i in range(Ms): MX_arrays.append([])

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

    # repair count for each PX (P01~P31)
    PX_arrays = []
    for i in range(Ps): PX_arrays.append([])

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
def kNN(aggregated, MX_corrcoef, PX_corrcoef):
    
    # read Output_TargetID_Mapping.csv
    mapping = pd.read_csv('Output_TargetID_Mapping.csv')
    mapping = np.array(mapping)

    result_to_submit = []
    explanation = []

    N = 10
    limit = 5

    for i in range(len(mapping)):
        print(i)

        map_i = mapping[i]
        
        val0_M = map_i[0]
        val0_P = map_i[1]
        val0_mo = map_i[3]
        val0_ym = int(map_i[2]) * 12 + val0_mo

        # compute distance
        dist_aggregated = []

        for j in range(len(aggregated)):

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
            repairSum += float(dist_aggregated[j][2]) / float(dist_aggregated[j][1])
            distanceSum += 1.0 / float(dist_aggregated[j][1])

            # write explanation
            # [val0_M, val0_P, val0_year, val0_month, rank,
            #  index, distance, repair count, val1_M, val1_P, val1_year, val1_month]
            explanation.append([val0_M, val0_P, map_i[2], val0_mo, j,
                                dist_aggregated[j][0], dist_aggregated[j][1],
                                dist_aggregated[j][2], dist_aggregated[j][3],
                                dist_aggregated[j][4], dist_aggregated[j][5],
                                dist_aggregated[j][6]])

        result_to_submit.append(repairSum / distanceSum)

        # write result periodically
        if (i + 1) % 500 == 0:
            print('writing result ...')
            pd.DataFrame(result_to_submit).to_csv('kNN_result_to_submit_' + str(i) + '.csv')
            pd.DataFrame(explanation).to_csv('kNN_explanation_' + str(i) + '.csv')

    # write final result
    result_to_submit = pd.DataFrame(result_to_submit)
    result_to_submit_with_limit = result_to_submit.clip(0, limit)
    explanation = pd.DataFrame(explanation)
    
    result_to_submit.to_csv('kNN_result_to_submit.csv')
    result_to_submit_with_limit.to_csv('kNN_result_to_submit_limit_' + str(limit) + '.csv')
    explanation.to_csv('kNN_explanation.csv')

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

    kNN(repairAggregated, MX_corrcoef, MP_corrcoef)
