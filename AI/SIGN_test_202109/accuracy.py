import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    # load prediction and ground truth
    test_output_to_compare = np.array(pd.read_csv('signTest_groundTruth.csv', index_col=0))
    prediction_to_compare = np.array(pd.read_csv('signTest_prediction.csv', index_col=0))

    test_rows = len(test_output_to_compare)

    # compute mean-square error and accuracy
    MSE = 0.0
    accurateRows = 0

    for i in range(test_rows):
        ground_truth = test_output_to_compare[i]
        pred = prediction_to_compare[i]
        
        MSE += mean_squared_error(ground_truth, pred)
        if np.argmax(ground_truth) == np.argmax(pred): accurateRows += 1

    MSE /= test_rows
    accurateRows /= test_rows
    
    print('MSE      = ' + str(MSE))
    print('accuracy = ' + str(accurateRows))
