import math
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import pandas as pd

if __name__ == '__main__':

    ground_truth = 'bert_train_output.csv'
    prediction = 'bert_valid_prediction_model_0.csv'
    
    # read output data file
    ground_truth = np.array(pd.read_csv(ground_truth, index_col=0))
    prediction = np.array(pd.read_csv(prediction, index_col=0))

    print('ground truth:')
    print(np.array(ground_truth[:15]))

    print('prediction:')
    print(np.array(prediction[:15]))

    # print ROC-AUC score
    print(round(roc_auc_score(ground_truth, prediction), 4))
