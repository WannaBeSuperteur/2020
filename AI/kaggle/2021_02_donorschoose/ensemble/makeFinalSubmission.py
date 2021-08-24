import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    finalPrediction = pd.read_csv('finalPrediction.csv', index_col=0)
    sampleSubmission = pd.read_csv('sample_submission.csv')

    # write final result
    final_result = []
    for i in range(len(finalPrediction)):
        final_result.append([finalPrediction.iloc[i, 0]])

    final_result = pd.DataFrame(final_result)
    final_result.columns = ['project_is_approved']
    final_result.index = np.array(sampleSubmission['id'])

    # need to write 'id' at (0, 0) of the data
    final_result.to_csv('finalSubmission.csv')
    
