import math
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import pandas as pd

def findROCAUC(ground_truth, prediction, verbose):

    prediction_fn = prediction
    
    # read output data file
    ground_truth = np.array(pd.read_csv(ground_truth, index_col=0))
    prediction = np.array(pd.read_csv(prediction, index_col=0))

    if verbose == True:
        print('\nground truth:')
        print(np.array(ground_truth[:15]))

        print('\nprediction:')
        print(np.array(prediction[:15]))

    # print ROC-AUC score
    print('\nfile name: ' + prediction_fn)
    print('ROC-AUC : ' + str(round(roc_auc_score(ground_truth, prediction), 6)))
    
if __name__ == '__main__':

    for i in range(6):
        ground_truth = 'bert_train_output.csv'
        prediction = 'bert_valid_prediction_model_' + str(i) + '.csv'
    
        findROCAUC(ground_truth, prediction, False)
