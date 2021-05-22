import sys
sys.path.insert(0, '../../../AI_BASE')

import math
import numpy as np
import readData as RD

if __name__ == '__main__':

    count = 1

    for i in range(count):
        if i == 0:
            answer_array = np.array(RD.loadArray('bert_valid_result_count_' + str(i) + '.txt'))[:, 5:6].astype(float)
        else:
            answer_array += np.array(RD.loadArray('bert_valid_result_count_' + str(i) + '.txt'))[:, 5:6].astype(float)

    answer_array /= count

    RD.saveArray('final_answer.txt', answer_array)
