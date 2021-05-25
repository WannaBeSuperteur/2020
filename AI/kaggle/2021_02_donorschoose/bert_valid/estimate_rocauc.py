import math
import numpy as np
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    predictionFileNameFormat = 'lightGBM_tv_valid_'
    answerFileName = 'train_valid_output_lgbm.txt'
    times = 4

    # get prediction and answer
    for i in range(times):
        predictionFileName = predictionFileNameFormat + str(i) + '.txt'

        if i == 0:
            prediction = np.array(RD.loadArray(predictionFileName)).astype(float).T[0]
        else:
            prediction += np.array(RD.loadArray(predictionFileName)).astype(float).T[0]

    prediction /= times

    answer = np.array(RD.loadArray(answerFileName)).astype(float).T[0]

    # print prediction, answer and ROC-AUC score
    print(prediction)
    print(answer)

    print(round(roc_auc_score(answer, prediction), 4))
