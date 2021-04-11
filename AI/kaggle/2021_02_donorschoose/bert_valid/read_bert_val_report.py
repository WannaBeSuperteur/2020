import sys
sys.path.insert(0, '../../../AI_BASE')

import math
import numpy as np
import readData as RD
import deepLearning_main as DL

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

def readResult(pred, real, num, count):

    # assertion
    assert(len(pred) == len(real))

    # extract values
    vals = []
    result = [['thrs', 'TP', 'TN', 'FP', 'FN', 'accu', 'correl', 'roc-auc'],
              ['-', '-', '-', '-', '-', '-',
               round(np.corrcoef(pred, real)[0][1], 4),
               round(roc_auc_score(real, pred), 4)]]
    
    for i in range(len(pred)):
        vals.append([pred[i], real[i]])

    for i in range(1, 250):
        threshold = count * round(1 - pow(0.95, i), 6)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        pred_binary = []
        real_binary = []

        for j in range(len(vals)):
            if vals[j][0] >= threshold and vals[j][1] == 1:
                TP += 1
                pred_binary.append(1)
                real_binary.append(1)
            elif vals[j][0] < threshold and vals[j][1] == 0:
                TN += 1
                pred_binary.append(0)
                real_binary.append(0)
            elif vals[j][0] >= threshold and vals[j][1] == 0:
                FP += 1
                pred_binary.append(1)
                real_binary.append(0)
            elif vals[j][0] < threshold and vals[j][1] == 1:
                FN += 1
                pred_binary.append(0)
                real_binary.append(1)

        # compute correlation
        try:
            corr = round(np.corrcoef(pred_binary, real_binary)[0][1], 4)
        except:
            corr = '-'

        # record result
        result.append([round(threshold, 4),
                       TP, TN, FP, FN,
                       round(float((TP+TN)/(TP+TN+FP+FN)), 4),
                       corr,
                       round(roc_auc_score(real_binary, pred_binary), 4)])

    # write result
    print(result)
    RD.saveArray('bert_val_report_' + str(num) + '.txt', result)

def avgv(valid_result, valid_answer, weights, n):
    weighted_valid_result = np.average(valid_result, axis=1, weights=weights)
    readResult(weighted_valid_result, valid_answer[:, 0], n)

if __name__ == '__main__':

    count = 4

    # read files
    valid_results = []
    
    for i in range(count):
        valid_results.append(np.array(RD.loadArray('bert_valid_result_count_' + str(i) + '.txt')).astype(float))
    
    valid_answer = np.array(RD.loadArray('bert_valid_rightAnswer.txt')).astype(float)

    # create valid prediction (sum of N valid predictions)
    valid_result = np.array(valid_results[0])
    for i in range(1, count): valid_result += np.array(valid_results[i])

    # save valid prediction
    RD.saveArray('valid_result_sum.txt', valid_result)

    # compare valid prediction with right answer
    for i in range(2, 3):
        readResult(valid_result[:, i], valid_answer[:, 0], i, count)

    # weighted average
    #avgv(valid_result, valid_answer, [0, 1, 1, 0, 0, 0], '12')
