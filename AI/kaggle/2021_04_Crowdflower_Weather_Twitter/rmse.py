import sys
sys.path.insert(0, '../../../AI_BASE')
sys.path.insert(1, '../')
import math
import numpy as np
import readData as RD
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    valid_prediction = np.array(RD.loadArray('bert_valid_result.txt')).astype(float)
    valid_answer = np.array(RD.loadArray('train_valid_output.txt')).astype(float)
    cols = 24
    avgRMSE = 0

    for col in range(cols):
        RMSE = math.sqrt(mean_squared_error(valid_answer[:, col], valid_prediction[:, col]))
        avgRMSE += RMSE
        print('RMSE of column ' + str(col) + ' = ' + str(RMSE))

    avgRMSE /= cols

    print('average RMSE = ' + str(avgRMSE))
