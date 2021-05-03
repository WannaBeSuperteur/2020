import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

if __name__ == '__main__':

    count_lightGBM = 4
    count_DecisionTree = 4
    count_deepLearning = 0
    count = 0

    # do not use both deepLearning and other algorithm(s)
    assert(count_deepLearning == 0 or count_lightGBM + count_DecisionTree == 0)

    fn_out = 'train_valid_output.txt'
    fl_out = RD.loadArray(fn_out)
    fl_preds = []

    # read file for lightGBM
    for i in range(count_lightGBM):
        count += 1

        fn_pred = 'lightGBM_tv_predict_' + str(i) + '.txt'
        fl_preds.append(RD.loadArray(fn_pred))

        rows = len(fl_out)

    # read file for Decision Tree
    for i in range(count_DecisionTree):
        count += 1

        fn_pred = 'DecisionTree_tv_predict_' + str(i) + '.txt'
        fl_preds.append(RD.loadArray(fn_pred))

        rows = len(fl_out)

    # read file for basic deep learning
    for i in range(count_deepLearning):
        count += 1
        
        fn = 'report_val_' + str(i) + '.txt'
        f = open(fn, 'r')
        fl = f.readlines()
        f.close()

        rows = len(fl)-9

    # extract values
    pred_array = []
    real_array = []

    # convert into numeric array
    fl_preds = np.array(fl_preds).astype(float).T[0]
        
    for i in range(rows):
        if count_deepLearning == 0:
            pred = float(sum(fl_preds[i]) / count)
            real = float(fl_out[i][0])                

        else:
            pred = float(fl[i].split('[')[2].split(']')[0])
            real = float(fl[i].split('[')[3].split(']')[0])

        pred_array.append(pred)
        real_array.append(real)

    pred_array = np.array(pred_array)
    real_array = np.array(real_array)

    for j in range(100):
        threshold = 0.50025 + j * 0.005

        true_positive = ((pred_array >= threshold) & (real_array >= threshold)).sum()
        false_positive = ((pred_array < threshold) & (real_array < threshold)).sum()
        accuracy = (true_positive + false_positive) / rows
            
        print('accuracy( ' + str(round(threshold, 4)) + ' ) = ' + str(round(accuracy, 6)))
            
    print('RMSE = ' + str(round(np.sqrt(np.mean((np.array(pred_array) - np.array(real_array))**2)), 6)))
    print('roc-auc = ' + str(round(roc_auc_score(real_array, pred_array), 6)))
