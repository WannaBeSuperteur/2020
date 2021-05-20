import sys
sys.path.insert(0, '../../../AI_BASE')
sys.path.insert(1, '../')

import math
import numpy as np
import readData as RD

if __name__ == '__main__':

    count = 4

    for i in range(count):
        if i == 0:
            answer_array = np.array(RD.loadArray('bert_valid_result_count_' + str(i) + '.txt'))[:, 2:3].astype(float)
        else:
            answer_array += np.array(RD.loadArray('bert_valid_result_count_' + str(i) + '.txt'))[:, 2:3].astype(float)

    answer_array /= count

    RD.saveArray('final_answer.txt', answer_array)
            
