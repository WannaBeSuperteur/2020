import numpy as np
import pandas as pd

# TMI : sum of the prediction from Jan 2010 to Jul 2011 using data_all = 29190.944
#       and average is 29190.944 / 4256 = 6.859

# prediction for Jan 2010 (sale+dif = 82) ~ Jul 2011 (sale+dif = 100)
def getPrediction(data):

    # convert to NumPy array
    data = np.array(data)
    
    numSaleMonth = 80
    numSaleRepairDif = 60
    
    prediction = np.zeros((19))

    # predict using sum of the value with sale+dif = (X+82) for month X = 0...18
    # where month 0 = Jan 2010, month 18 = Jul 2011
    for i in range(numSaleMonth):
        for j in range(numSaleRepairDif):
            
            sumSaleDif = i + j

            if sumSaleDif >= 82 and sumSaleDif <= 100:
                prediction[sumSaleDif - 82] += data[i][j]

    return prediction

# add weighted sum using weighted array
def addWeightedSum(data, yearMonthIndex, weightedArray):

    # make the sum of weighted array to 1
    weightedArray = np.array(weightedArray).astype(float)
    sumWeights = sum(weightedArray)
    weightedArray /= sumWeights
    
    # add weighted sum
    result = 0.0
    available_weight_sum = 0.0
    midIndex = int((len(weightedArray) - 1) / 2)

    for i in range(len(weightedArray)):
        ymIndex = yearMonthIndex + i - midIndex
        
        if ymIndex >= 0 and ymIndex < len(data):
            result += data[yearMonthIndex + i - midIndex] * weightedArray[i]
            available_weight_sum += weightedArray[i]

    return result / available_weight_sum

# make submission using prediction by getPrediction(data)
# dataset : [data_all] or
#           [data_M1, data_M2, ..., data_M9] or
#           [data_P01, data_P02, ..., data_P09]
def makeSubmission(dataset, option):

    Ms = 9 # M1~M9
    Ps = 31 # P01~P31

    mapping = pd.read_csv('Output_TargetID_Mapping.csv')
    print(mapping)

    submission_count = 4256
    final_submission = np.zeros((submission_count))

    # make final submission
    for i in range(submission_count):

        M_index = int(mapping.iloc[i][0][1:]) - 1
        P_index = int(mapping.iloc[i][1][1:]) - 1
        
        year = mapping.iloc[i][2]
        month = mapping.iloc[i][3]
        yearMonthIndex = (year - 2010) * 12 + (month - 1)

        # option == 'all' -> sum(dataset) / (Ms * Ps) for each month
        if option == 'all':
            final_submission[i] += addWeightedSum(dataset[0], yearMonthIndex,
                                                  [1, 2, 3, 4, 5, 4, 3, 2, 1]) / (Ms * Ps)

        # option == 'M' -> sum(dataset with MX) / Ps for each month
        elif option == 'M':

            # find corresponding data with MX from the dataset
            for j in range(Ms):
                if M_index == j:
                    final_submission[i] = addWeightedSum(dataset[j], yearMonthIndex,
                                                         [1, 2, 3, 4, 5, 4, 3, 2, 1]) / Ps

        # option == 'P' -> sum(dataset with PX) / Ms for each month
        elif option == 'P':

            # find corresponding data with PX from the dataset
            for j in range(Ps):
                if P_index == j:
                    final_submission[i] = addWeightedSum(dataset[j], yearMonthIndex,
                                                         [1, 2, 3, 4, 5, 4, 3, 2, 1]) / Ms

    return pd.DataFrame(final_submission)

if __name__ == '__main__':

    Ms = 9 # M1~M9
    Ps = 31 # P01~P31

    # prediction about (ALL)
    data_all = pd.read_csv('predict_all_filled.csv', index_col=0)

    # prediction about M1~M9
    data_Ms = []
    for i in range(Ms):
        data_Ms.append(pd.read_csv('predict_M' + str(i+1) + '_filled.csv', index_col=0))

    # prediction about P01~P31
    data_Ps = []
    for i in range(Ps):
        data_Ps.append(pd.read_csv('predict_P' + '%02d' % (i+1) + '_filled.csv', index_col=0))

    # final submission about (ALL)
    prediction_all = getPrediction(data_all)
    final_submission_all = makeSubmission([prediction_all], 'all')
    final_submission_all.to_csv('final_submission_all.csv')

    # final submission about M1~M9
    prediction_Ms = []
    for i in range(Ms):
        prediction_Ms.append(getPrediction(data_Ms[i]))
        
    final_submission_Ms = makeSubmission(prediction_Ms, 'M')
    final_submission_Ms.to_csv('final_submission_Ms.csv')

    # final submission about P01~P31
    prediction_Ps = []
    for i in range(Ps):
        prediction_Ps.append(getPrediction(data_Ps[i]))
        
    final_submission_Ps = makeSubmission(prediction_Ps, 'P')
    final_submission_Ps.to_csv('final_submission_Ps.csv')

    # final submission using average
    fs_all = np.array(final_submission_all)
    fs_Ms = np.array(final_submission_Ms)
    fs_Ps = np.array(final_submission_Ps)
    
    final_submission_avg = pow(fs_all * fs_Ms * fs_Ps, 1/3) - 1.0
    final_submission_avg = np.clip(final_submission_avg, 0.0, 1.0)
    final_submission_avg = pd.DataFrame(final_submission_avg)
    
    final_submission_avg.to_csv('final_submission_avg.csv')
