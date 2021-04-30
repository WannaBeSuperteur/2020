import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    count = 1

    for i in range(count):

        lightGBM = True

        # read file
        if lightGBM == True:
            fn_out = 'lightGBM_tv_output_' + str(i) + '.txt'
            fn_pred = 'lightGBM_tv_predict_' + str(i) + '.txt'
            
            f_out = open(fn_out, 'r')
            f_pred = open(fn_pred, 'r')
            
            fl_out = f_out.readlines()
            fl_pred = f_pred.readlines()
            
            f_out.close()
            f_pred.close()

            rows = len(fl_out)-9

        else:
            fn = 'report_val_' + str(i) + '.txt'
            f = open(fn, 'r')
            fl = f.readlines()
            f.close()

            rows = len(fl)-9

        # extract values
        pred_array = []
        real_array = []
        
        for i in range(rows):
            if lightGBM == True:
                pred = float(fl_pred[i])
                real = float(fl_out[i])                

            else:
                pred = float(fl[i].split('[')[2].split(']')[0])
                real = float(fl[i].split('[')[3].split(']')[0])

            pred_array.append(pred)
            real_array.append(real)

        pred_array = np.array(pred_array)
        real_array = np.array(real_array)

        for j in range(100):
            threshold = 0.005 + j * 0.01

            true_positive = ((pred_array >= threshold) & (real_array >= threshold)).sum()
            false_positive = ((pred_array < threshold) & (real_array < threshold)).sum()
            accuracy = (true_positive + false_positive) / rows
            
            print('accuracy( ' + str(round(threshold, 4)) + ' ) = ' + str(round(accuracy, 6)))
            
        print('RMSE = ' + str(round(np.sqrt(np.mean((np.array(pred_array) - np.array(real_array))**2)), 6)))
        print('roc-auc = ' + str(round(roc_auc_score(real_array, pred_array), 6)))
