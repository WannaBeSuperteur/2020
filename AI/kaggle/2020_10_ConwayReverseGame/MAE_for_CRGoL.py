import math
import numpy as np

# file input and output
# input  : validation report
# output : validation report with weighted-average for each cell
#          validation MAE with changing threshold

# files read by this code
# valid_report_n_sub_X.txt (open in readValidReport)

# files written by this code
# valid_report_n_sub_X_weightedAvg.txt (open in readValidReport)
# valid_report_n_sub_X_MAE.txt (open in readValidReport)

# board_i_y and board_i_x -> index of the data
def xyToIndex(board_i_y, board_i_x, i, size):
    i_start = int(i / (size*size)) * (size*size)
    return i_start + board_i_y * size + board_i_x

# available for BINARY (0 or 1) values only
# fn            : file name
# thresholdList : list for thresholds (above->1, below->0)
# size          : input board size (width or height = 20)
# n             : number of values in each array pred = [...] or real = [...]
def readValidReport(fn, thresholdList, size, n):

    # number of rows in each output array (rows * rows board)
    rows = int(math.sqrt(n)) # number of rows
    outputWS = int((rows-1)/2) # output window size

    # read validation report
    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # read each line and compute MAE
    MAE = [] # MAE values (for each threshold)
    zeros = [] # count of number 0
    ones = [] # count of number 1
    
    for i in range(len(thresholdList)):
        MAE.append(0.0)
        zeros.append(0.0)
        ones.append(0.0)
        
    lines = len(fl)-9 # number of lines

    # weight for each cell (left-top ~ right-bottom of output board)
    weight = [1, 1, 1, 1, 2, 1, 1, 1, 1]
    weightedAvgResult = str(weight) + '\n' # to write valid_report_n_sub_X_weightedAvg.txt

    # center of output vector
    center = int((n-1)/2)
    
    for i in range(lines):
        if i % 50 == 0: print(i)

        # index of the line(indexed by i) for the board
        board_i = i % (size * size)
        board_i_y = int(board_i / size) # y-axis value for the board
        board_i_x = board_i % size # x-axis value for the board
        
        # read predicted and real values
        pred = fl[i].split('pred = [')[1].split(']')[0] # pred = [...]
        real = fl[i].split('real = [')[1].split(']')[0] # real = [...]

        # convert to float
        predSplit = np.array(' '.join(pred.split()).split(' ')).astype('float')
        realSplit = np.array(' '.join(real.split()).split(' ')).astype('float')

        # computed prediction using weighted average (weight: defined in array 'weight')
        weightedPred = 0.0
        sumWeight = 0.0

        # to index weight array
        y_start = board_i_y - outputWS
        x_start = board_i_x - outputWS

        refList = []
        refVals = []
        refWeights = []
        
        for j in range(max(0, board_i_y - outputWS), min(size, board_i_y + outputWS + 1)):
            for k in range(max(0, board_i_x - outputWS), min(size, board_i_x + outputWS + 1)):

                # read predicted output of this index
                outputIndex = xyToIndex(j, k, i, size)

                pred_ = fl[outputIndex].split('pred = [')[1].split(']')[0]
                predSplit_ = np.array(' '.join(pred_.split()).split(' ')).astype('float')

                # update weighted prediction
                refPosition = (j - y_start) * rows + (k - x_start) # position of the location width index (j, k) in array 'weight'
                
                weightedPred += weight[refPosition] * predSplit_[n - 1 - refPosition]
                sumWeight += weight[refPosition]

                refList.append(outputIndex)
                refVals.append(round(predSplit_[n - 1 - refPosition], 6))
                refWeights.append(weight[refPosition])

        # divide by sum of the weights and use it as prediction
        weightedPred /= sumWeight
        weightedAvgResult += ('[' + str(i) + '] refList=' + str(refList) + ' refVals=' + str(refVals) + ' refWeights=' + str(refWeights)
                              + ' weightedPred=' + str(round(weightedPred, 6)) + '\n')

        # compute MAE for each threshold
        for j in range(len(thresholdList)):
            
            # pred is considered as 1 -> MAE=0 if real=1, MAE=1 if real=0
            if weightedPred > thresholdList[j]:
                MAE[j] += (1 - int(realSplit[center]))
                ones[j] += 1.0

            # pred is considered as 0 -> MAE=0 if real=0, MAE=1 if real=1
            else:
                MAE[j] += int(realSplit[center])
                zeros[j] += 1.0

    # MAE <- MAE / (lines * n)
    # zeros,ones <- zeros,ones / (number of total count)
    for i in range(len(thresholdList)):

        # for MAE, divide by the number of rows
        MAE[i] = MAE[i] / lines

        # for zeros and ones, divide to make the sum 1.0
        zeros_ = zeros[i]
        ones_ = ones[i]
        zeros[i] = zeros[i] / (zeros_ + ones_)
        ones[i] = ones[i] / (zeros_ + ones_)
        
    # print MAE
    result = '' # to write valid_report_n_sub_X_MAE.txt
    for i in range(len(thresholdList)):
        result_ = ('MAE when threshold = ' + str(round(thresholdList[i], 6)) + '\tis ' + str(round(MAE[i], 6))
                   + '\t0 in pred = ' + str(round(zeros[i], 6)) + ',\t1 in pred = ' + str(round(ones[i], 6)))
        print(result_)
        result += result_ + '\n'

    # write to file
    f = open(fn[:len(fn)-4] + '_MAE.txt', 'w')
    f.write(result)
    f.close()

    f = open(fn[:len(fn)-4] + '_weightedAvg.txt', 'w')
    f.write(weightedAvgResult)
    f.close()

if __name__ == '__main__':
    thresholdList = []
    for i in range(1, 100): thresholdList.append(round(0.01*i, 6))
    readValidReport('valid_report_n_sub_0.txt', thresholdList, 20, 3*3)
    readValidReport('valid_report_n_sub_1.txt', thresholdList, 20, 3*3)
    readValidReport('valid_report_n_sub_2.txt', thresholdList, 20, 3*3)
    readValidReport('valid_report_n_sub_3.txt', thresholdList, 20, 3*3)
    readValidReport('valid_report_n_sub_4.txt', thresholdList, 20, 3*3)
