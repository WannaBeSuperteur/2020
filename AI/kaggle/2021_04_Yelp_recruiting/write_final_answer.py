# open sample_submission.csv refer to
# https://stackoverflow.com/questions/53410490/i-am-getting-an-error-as-name-while-opening-csv-file-in-excel-2016

import numpy as np
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

if __name__ == '__main__':

    # final result
    finalResult = ''

    # read file
    result_funny = RD.loadArray('test_output_funny.txt')
    result_useful = RD.loadArray('test_output_useful.txt')
    result_cool = RD.loadArray('test_output_cool.txt')
    results = 22956

    # avg and std for funny / useful / cool
    avgs_and_stds = RD.loadArray('train_output_avg_and_std.txt')
    print(avgs_and_stds)

    for i in range(results):

        # sum of funny, useful and cool
        funny = float(result_funny[i][0]) * float(avgs_and_stds[1][0]) + float(avgs_and_stds[0][0])
        useful = float(result_useful[i][0]) * float(avgs_and_stds[1][1]) + float(avgs_and_stds[0][1])
        cool = float(result_cool[i][0]) * float(avgs_and_stds[1][2]) + float(avgs_and_stds[0][2])

        # non-negative
        finalResult += str(max(0, funny + useful + cool)) + '\n'

    f = open('final_result.txt', 'w')
    f.write(finalResult)
    f.close()
