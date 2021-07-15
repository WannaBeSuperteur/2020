import sys
import math
import numpy as np
import pandas as pd

if __name__ == '__main__':

    count = 1

    for i in range(count):
        bert_result = pd.read_csv('bert_valid_result_count_' + str(i) + '.csv', index_col=0)
        
        if i == 0:
            answer_array = np.array(bert_result)[:, 2:3].astype(float)
        else:
            answer_array += np.array(bert_result)[:, 2:3].astype(float)

    answer_array /= count

    pd.DataFrame(answer_array).to_csv('final_answer.csv')
