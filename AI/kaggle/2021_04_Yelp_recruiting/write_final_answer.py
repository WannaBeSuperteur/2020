# open sample_submission.csv refer to
# https://stackoverflow.com/questions/53410490/i-am-getting-an-error-as-name-while-opening-csv-file-in-excel-2016

import numpy as np
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

if __name__ == '__main__':

    # final result
    finalResult = 'Id,Votes\n'

    sampleSub = RD.loadArray('sample_submission.csv', ',')
    
    results = 22956
    times = 16
    algorithm = 'deepLearning'

    # avg and std for votes
    avgs_and_stds = RD.loadArray('train_output_avg_and_std.txt')
    print(avgs_and_stds)

    # sum of prediction of useful votes for each review
    final_sum = []

    # read file
    for count in range(times):

        print('count = ' + str(count))

        if algorithm == 'deepLearning':
            result = RD.loadArray('test_output_' + str(count) + '.txt')
        else:
            result = RD.loadArray('lightGBM_test_result_' + str(count) + '.txt')

        for i in range(results):

            # predicted number of useful votes
            votes = float(result[i][0]) * float(avgs_and_stds[0][1]) + float(avgs_and_stds[0][0])
            if i < 10: print(votes)

            if count == 0:
                final_sum.append(votes)
            else:
                final_sum[i] += votes

    # write final prediction of useful votes as non-negative
    # using average of useful votes for each review
    for i in range(results):
        finalResult += sampleSub[i+1][0].split(',')[0] + ',' + str(max(0, final_sum[i] / times)) + '\n'

    f = open('final_sample_submission.csv', 'w')
    f.write(finalResult)
    f.close()
