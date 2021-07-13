import math
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import pandas as pd

# convert value
def convert(value, arr):

    if value == 1.0: return 1.0
    if value == 0.0: return 0.0

    arrLen = len(arr) - 1

    # when value=0.73, arr=[0.0, 0.5, 0.7, 0.85, 0.95, 1.0], arrLen=5
    # -> loc   = value * arrLen = 0.73 * 5 = 3.65
    loc = value * arrLen
    
    # -> left  = arr[int(3.65)]   = 0.85
    # -> right = arr[int(3.65)+1] = 0.95
    # -> rate  = 3.65 % 1.0       = 0.65
    left = arr[int(loc)]
    right = arr[int(loc) + 1]
    rate = loc % 1.0

    # to return: 0.85 + (0.95 - 0.85) * 0.65 = 0.915
    result = left + (right - left) * rate
    return result

if __name__ == '__main__':

    outputFileName = 'bert_valid_result.txt'
    saveFileName = 'bert_valid_result_converted.txt'
    arr = [0.0, 0.01, 0.03, 0.06, 0.1, 0.18, 0.32, 0.5, 0.68, 0.82, 0.9, 0.94, 0.97, 0.99, 1.0]

    # read output data file
    outputData = pd.read_csv(outputFileName, index_col=0)

    for i in range(len(outputData)):
        for j in range(len(outputData[i])):
            outputData[i][j] = convert(float(outputData[i][j]), arr)

    # save result
    RD.saveArray(saveFileName, outputData, '\t', 500)

    # print ROC-AUC score
    converted = np.array(pd.read_csv(saveFileName, index_col=0)).astype(float)
    valid_answer = np.array(pd.read_csv('bert_valid_rightAnswer.txt', index_col=0)).astype(float)

    pred_array = converted[:, 2]
    real_array = valid_answer[:, 0]

    print(pred_array)
    print(real_array)

    print(round(roc_auc_score(real_array, pred_array), 4))
