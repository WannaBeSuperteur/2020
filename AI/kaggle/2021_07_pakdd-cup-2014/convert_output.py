import numpy as np
import pandas as pd
import math

# inverse of sigmoid
def invSigmoid(value):
    return math.log(value / (1.0 - value))

if __name__ == '__main__':
    
    avgAndStd = np.array(pd.read_csv('train_meanAndStd.csv', index_col=0))

    countLog_mean = avgAndStd[2][0]
    countLog_std = avgAndStd[2][1]

    final_prediction = np.array(pd.read_csv('final_prediction.csv', index_col=0))

    # Y -> log2(Y+1) -> normalize(log2(Y+1)) -> sigmoid(normalize(log2(Y+1)))
    for i in range(len(final_prediction)):

        # invSigmoid
        final_prediction[i][0] = invSigmoid(final_prediction[i][0])

        # denormalize
        final_prediction[i][0] = final_prediction[i][0] * countLog_std + countLog_mean

        # log2
        final_prediction[i][0] = pow(2, final_prediction[i][0]) - 1
        
    pd.DataFrame(final_prediction).to_csv('final_prediction_converted.csv')
