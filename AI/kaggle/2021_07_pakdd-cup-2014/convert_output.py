import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    avgAndStd = np.array(pd.read_csv('train_meanAndStd.csv', index_col=0))

    countLog_mean = avgAndStd[2][0]
    countLog_std = avgAndStd[2][1]

    final_prediction = np.array(pd.read_csv('final_prediction.csv', index_col=0))

    for i in range(len(final_prediction)):
        final_prediction[i][0] = pow(2, (final_prediction[i][0] - countLog_mean)/countLog_std)
        
    pd.DataFrame(final_prediction).to_csv('final_prediction_converted.csv')
