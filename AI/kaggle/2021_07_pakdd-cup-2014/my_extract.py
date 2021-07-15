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

    # aggregate
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
    train_X = np.zeros((agg_len, modules + components + 2))
    train_Y = np.zeros((agg_len, 1))

    for i in range(agg_len):

        X_split = agg_data[i][0].split(' ')

        # module
        train_X[i][int(X_split[0][1:]) - 1] = 1

        # component
        train_X[i][modules + int(X_split[1][1:]) - 1] = 1

        # repair
        train_X[i][modules + components] = yearMonthToInt(X_split[3])

        # repair (month)
        train_X[i][modules + components + 1] = int(X_split[3].split('/')[1])

        # count (apply base 2 log)
        train_Y[i][0] = math.log(int(agg_data[i][1]) + 1, 2)

    # find avg and stddev
    repair_mean      = train_X.mean(axis=0)[modules + components]
    repairMonth_mean = train_X.mean(axis=0)[modules + components + 1]
    countLog_mean    = train_Y.mean(axis=0)[0]

    repair_std       = train_X.std(axis=0)[modules + components]
    repairMonth_std  = train_X.std(axis=0)[modules + components + 1]
    countLog_std     = train_Y.std(axis=0)[0]

    meanAndStd = np.array([[repair_mean, repair_std],
                           [repairMonth_mean, repairMonth_std],
                           [countLog_mean, countLog_std]])

    pd.DataFrame(meanAndStd).to_csv('train_meanAndStd.csv')

    # convert using avg and stddev
    for i in range(agg_len):
        train_X[i][modules + components]     = (train_X[i][modules + components] - repair_mean)          / repair_std
        train_X[i][modules + components + 1] = (train_X[i][modules + components + 1] - repairMonth_mean) / repairMonth_std
        train_Y[i][0]                        = (train_Y[i][0] - countLog_mean)                           / countLog_std

    pd.DataFrame(train_X).to_csv('train_X.csv')
    pd.DataFrame(train_Y).to_csv('train_Y.csv')

# create test_X data using output target ID mapping
def createTestX(outputTargetIDMapping):

    test_X_rows = len(outputTargetIDMapping)
    test_X = np.zeros((test_X_rows, modules + components + 2))

    for i in range(test_X_rows):

        # module
        test_X[i][int(outputTargetIDMapping[i][0][1:]) - 1] = 1

        # component
        test_X[i][modules + int(outputTargetIDMapping[i][1][1:]) - 1] = 1

        # repair
        test_X[i][modules + components] = yearAndMonthToInt(outputTargetIDMapping[i][2], outputTargetIDMapping[i][3])

        # repair (month)
        test_X[i][modules + components + 1] = math.log(int(outputTargetIDMapping[i][3]), 2)

    # normalize using avg and stddev
    avgAndStd = np.array(pd.read_csv('train_meanAndStd.csv', index_col=0))

    repair_mean      = avgAndStd[0][0]
    repairMonth_mean = avgAndStd[1][0]
    countLog_mean    = avgAndStd[2][0]

    repair_std       = avgAndStd[0][1]
    repairMonth_std  = avgAndStd[1][1]
    countLog_std     = avgAndStd[2][1]

    # convert using avg and stddev
    for i in range(test_X_rows):
        test_X[i][modules + components]     = (test_X[i][modules + components] - repair_mean)          / repair_std
        test_X[i][modules + components + 1] = (test_X[i][modules + components + 1] - repairMonth_mean) / repairMonth_std
        
    pd.DataFrame(test_X).to_csv('test_X.csv')

if __name__ == '__main__':

    modules = 9
    components = 31

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
    
    # create test_X data using output target ID mapping
    createTestX(np.array(outputTargetIDMapping))
