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
        aggregated[i][1] = 'P' + str(i // repairYM % Ps + 1)
        aggregated[i][2] = str((i % repairYM + 2004 * 12 + 1) // 12 + 1)
        aggregated[i][3] = (i % repairYM + 1) % 12 + 1

    print(aggregated)
    pd.DataFrame(aggregated).to_csv('test.csv')

# compure correlation coefficient for each PX or MX
def computeCorrCoef():
    
    pass

# kNN for [MX, PX, year(repair), month(repair)]
def kNN():
    pass

# using Output_TargetID_Mapping.csv and kNN to make final submission
def predictForSubmission():
    pass

# 0. aggregate repair count : group by (MX, PX, year(repair), month(repair))
# 1. compute correlation coefficient CORR for each MX, using repair count for each year(repair), month(repair) and PX
#    compute correlation coefficient CORR for each PX, using repair count for each year(repair), month(repair) and MX
# 2. using kNN by module type(MX), component type(PX), year(repair), month(repair) and year*12+month(repair)
#    using (1 - CORR) for different MX and PX, normalized distance for other variables
if __name__ == '__main__':
    
    repairAggregated = aggregateRepairCount()

    
