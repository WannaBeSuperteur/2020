import sys
import math
import numpy as np
import pandas as pd

if __name__ == '__main__':

    bert_result = pd.read_csv('bert_test_prediction.csv', index_col=0)
    answer_array = np.array(bert_result)[:, 2:3].astype(float)

    pd.DataFrame(answer_array).to_csv('final_answer.csv')
