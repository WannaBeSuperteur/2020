import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

def getPredAndRealArray(count_lightGBM, count_DecisionTree, count_XGBoost, count_deepLearning, fn_out, valid):
    
    count = 0

    # do not use both deepLearning and other algorithm(s)
    if fn_out != None:
        fl_out = RD.loadArray(fn_out)

    fl_preds = []

    # read file for lightGBM
    for i in range(count_lightGBM):
        count += 1

        fn_pred = 'lightGBM_tv_predict_' + str(i) + '.txt'
        fl_preds.append(RD.loadArray(fn_pred))

    # read file for Decision Tree
    for i in range(count_DecisionTree):
        count += 1

        fn_pred = 'DecisionTree_tv_predict_' + str(i) + '.txt'
        fl_preds.append(RD.loadArray(fn_pred))

    # read file for XGBoost
    for i in range(count_XGBoost):
        count += 1

        fn_pred = 'XGBoost_tv_predict_' + str(i) + '.txt'
        fl_preds.append(RD.loadArray(fn_pred))

    # read file for basic deep learning
    for i in range(count_deepLearning):
        count += 1
        
        if valid == True:
            fn_pred = 'train_valid_predict_' + str(i) + '.txt'
        else:
            fn_pred = 'test_predict_' + str(i) + '.txt'

        fl_preds.append(np.array(RD.loadArray(fn_pred))[:, 0:1])

    # extract values
    pred_array = []
    if fn_out != None: real_array = []

    # convert into numeric array
    fl_preds = np.array(fl_preds).astype(float).T[0]

    rows = len(fl_preds)
        
    for i in range(rows):
        pred = float(sum(fl_preds[i]) / count)
        if fn_out != None: real = float(fl_out[i][0])                

        pred_array.append(pred)
        if fn_out != None: real_array.append(real)

    pred_array = np.array(pred_array)
    if fn_out != None: real_array = np.array(real_array)

    if fn_out == None:
        return pred_array
    else:
        return (rows, pred_array, real_array)

if __name__ == '__main__':

    count_lightGBM     = [1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 1, 2, 4, 1, 2, 4]
    count_DecisionTree = [0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 2, 4]
    count_XGBoost      = [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 1, 2, 4, 1, 2, 4]
    count_deepLearning = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 1, 2, 4, 1, 2, 4]
    
    fn_out = 'train_valid_output.txt'
    valid = True

    assert(len(count_lightGBM) == len(count_DecisionTree))
    assert(len(count_lightGBM) == len(count_XGBoost))
    assert(len(count_lightGBM) == len(count_deepLearning))

    # count_lightGBM, count_DecisionTree, count_XGBoost, count_deepLearning
    # max_accuracy, max_accuracy_at, RMSE and roc_auc
    info = []

    for i in range(len(count_lightGBM)):

        # max accuracy
        (rows, pred_array, real_array) = getPredAndRealArray(count_lightGBM[i],
                                                             count_DecisionTree[i],
                                                             count_XGBoost[i],
                                                             count_deepLearning[i], fn_out, valid)

        max_accuracy = 0
        max_accuracy_at = 0.0

        for j in range(100):
            threshold = 0.005 + j * 0.01

            true_positive = ((pred_array >= threshold) & (real_array >= threshold)).sum()
            false_positive = ((pred_array < threshold) & (real_array < threshold)).sum()
            accuracy = (true_positive + false_positive) / rows
                
            print('accuracy( ' + str(round(threshold, 4)) + ' ) = ' + str(round(accuracy, 6)))

            # update max_accuracy
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_at = threshold

        # RMSE and roc_auc
        RMSE = np.sqrt(np.mean((np.array(pred_array) - np.array(real_array))**2))
        roc_auc = roc_auc_score(real_array, pred_array)
        
        print('RMSE = ' + str(round(RMSE, 6)))
        print('roc-auc = ' + str(round(roc_auc, 6)))

        info.append([count_lightGBM[i], count_DecisionTree[i], count_XGBoost[i], count_deepLearning[i],
                     round(max_accuracy, 4), round(max_accuracy_at, 4), round(RMSE, 4), round(roc_auc, 4)])

    # save info
    RD.saveArray('valid_info.txt', info)
