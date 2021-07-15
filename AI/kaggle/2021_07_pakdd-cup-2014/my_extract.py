import pandas as pd
import numpy as np
import math

# 2007/10 -> 2007*12 + (10 - 1)
def yearMonthToInt(yearMonth):
    year = int(yearMonth.split('/')[0])
    month = int(yearMonth.split('/')[1])

    return yearAndMonthToInt(year, month)

# [2007, 10] -> 2007*12 + (10 - 1)
def yearAndMonthToInt(year, month):
    return year * 12 + (month - 1)

# preprocess RepairTrain.csv
def preprocess(repairTrain):

    print('\n\npreprocessing...\n\n')

    rows = len(repairTrain)
    print(rows)
    
    repairTrainArray = np.array(repairTrain)
    print(repairTrainArray)

    # module_category    : one-hot     (9)
    # component_category : one-hot     (31)
    # year               : avg and std (5)
    # month              : avg and std (12)
    # year*12+(month-1)  : avg and std (59)
    # so 9 * 31 * 59 = 16,461 records needed for aggregation

    agg_dic = {}

    for row in range(rows):

        module = repairTrainArray[row][0]
        component = repairTrainArray[row][1]
        sale = repairTrainArray[row][2]
        repair = repairTrainArray[row][3]
        count = repairTrainArray[row][4]

        agg_name = module + ' ' + component + ' ' + sale + ' ' + repair

        try:
            agg_dic[agg_name] += count
        except:
            agg_dic[agg_name] = count

    # convert dict to array
    agg_dic_x = list(agg_dic.keys())
    agg_dic_y = list(agg_dic.values())
    agg_data = np.vstack([agg_dic_x, agg_dic_y]).T
    agg_len = len(agg_data)
    
    print(agg_len)
    print(np.array(agg_data))

    # convert this array into preprocessed array
    modules = 9
    components = 31
    
    train_X = np.zeros((agg_len, modules + components + 4))
    train_Y = np.zeros((agg_len, 1))

    for i in range(agg_len):

        X_split = agg_data[i][0].split(' ')

        # module
        train_X[i][int(X_split[0][1:]) - 1] = 1

        # component
        train_X[i][modules + int(X_split[1][1:]) - 1] = 1

        # sale
        train_X[i][modules + components] = yearMonthToInt(X_split[2])

        # sale (month)
        train_X[i][modules + components + 1] = int(X_split[2].split('/')[1])

        # repair
        train_X[i][modules + components + 2] = yearMonthToInt(X_split[3])

        # repair (month)
        train_X[i][modules + components + 3] = int(X_split[3].split('/')[1])

        # count (apply base 2 log)
        train_Y[i][0] = math.log(int(agg_data[i][1]) + 1, 2)

    # find avg and stddev
    sale_mean        = train_X.mean(axis=0)[modules + components]
    saleMonth_mean   = train_X.mean(axis=0)[modules + components + 1]
    repair_mean      = train_X.mean(axis=0)[modules + components + 2]
    repairMonth_mean = train_X.mean(axis=0)[modules + components + 3]
    countLog_mean    = train_Y.mean(axis=0)[0]

    sale_std         = train_X.std(axis=0)[modules + components]
    saleMonth_std    = train_X.std(axis=0)[modules + components + 1]
    repair_std       = train_X.std(axis=0)[modules + components + 2]
    repairMonth_std  = train_X.std(axis=0)[modules + components + 3]
    countLog_std     = train_Y.std(axis=0)[0]

    meanAndStd = np.array([[sale_mean, sale_std],
                           [saleMonth_mean, saleMonth_std],
                           [repair_mean, repair_std],
                           [repairMonth_mean, repairMonth_std],
                           [countLog_mean, countLog_std]])

    pd.DataFrame(meanAndStd).to_csv('train_meanAndStd.csv')

    # convert using avg and stddev
    for i in range(agg_len):
        train_X[i][modules + components]     = (train_X[i][modules + components] - sale_mean)            / sale_std
        train_X[i][modules + components + 1] = (train_X[i][modules + components + 1] - saleMonth_mean)   / saleMonth_std
        train_X[i][modules + components + 2] = (train_X[i][modules + components + 2] - repair_mean)      / repair_std
        train_X[i][modules + components + 3] = (train_X[i][modules + components + 3] - repairMonth_mean) / repairMonth_std
        train_Y[i][0]                        = (train_Y[i][0] - countLog_mean)                           / countLog_std

    pd.DataFrame(train_X).to_csv('train_X.csv')
    pd.DataFrame(train_Y).to_csv('train_Y.csv')

if __name__ == '__main__':

    desireOutputRef = pd.DataFrame(np.array(pd.read_csv('DesireOutputRef.csv')))
    fixTrain = pd.DataFrame(np.array(pd.read_csv('FixTrain.csv')))
    repairTrain = pd.read_csv('RepairTrain.csv')
    saleTrain = pd.read_csv('SaleTrain.csv')
    outputTargetIDMapping = pd.read_csv('Output_TargetID_Mapping.csv')

    print('\n\ndesire output ref:')
    print(desireOutputRef)

    print('\n\nfix train:')
    print(fixTrain)

    print('\n\nrepair train:')
    print(repairTrain)

    print('\n\nsale train:')
    print(saleTrain)

    print('\n\noutput target ID mapping:')
    print(outputTargetIDMapping)

    # preprocess repair-train data
    preprocess(repairTrain)
    
