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

# get weighted prediction
# n               : number of cells in test output array ( = rows * rows = len(weight))
# i               : index
# size            : board size (25 for this problem)
# board_i_y       : y-axis value of index i of the board
# board_i_x       : x-axis value of index i of the board
# weight          : weight values
# readValidReport : reading valid report (True or False)
# fl              : file.readlines() (valid report : valid_report_n_sub_X.txt, or test output : test_output_n_sub_X.txt)
def getWeightedPrediction(n, i, size, board_i_y, board_i_x, weight, readValidReport, fl):

    # compute prediction using weighted average (weight: defined in array 'weight')
    weightedPred = 0.0
    sumWeight = 0.0

    # number of rows in each output array (n = rows * rows board)
    rows = int(math.sqrt(n)) # number of rows in test output
    outputWS = int((rows-1)/2) # output window size
    
    y_start = board_i_y - outputWS
    x_start = board_i_x - outputWS

    refList = []
    refVals = []
    refWeights = []
        
    for j in range(max(0, board_i_y - outputWS), min(size, board_i_y + outputWS + 1)):
        for k in range(max(0, board_i_x - outputWS), min(size, board_i_x + outputWS + 1)):

            # read predicted output of this index
            outputIndex = xyToIndex(j, k, i, size)

            # find output values (= valid/test prediction)
            if readValidReport == True: # valid prediction output
                pred_ = fl[outputIndex].split('pred = [')[1].split(']')[0]
                output = np.array(' '.join(pred_.split()).split(' ')).astype('float')
            else: # test prediction output
                
                # result value array (with output size)
                output = fl[outputIndex].split('\n')[0].split('\t')
                output = output[:len(output)-1]
                output = np.array(output).astype('float')

            # update weighted prediction
            refPosition = (j - y_start) * rows + (k - x_start) # position of the location width index (j, k) in array 'weight'
             
            weightedPred += weight[refPosition] * output[n - 1 - refPosition]
            sumWeight += weight[refPosition]

            refList.append(outputIndex)
            refVals.append(round(output[n - 1 - refPosition], 6))
            refWeights.append(weight[refPosition])

    return (weightedPred / sumWeight, refList, refVals, refWeights)

# available for BINARY (0 or 1) values only
# fn            : file name
# thresholdList : list for thresholds (above->1, below->0)
# size          : input board size (width or height = 25)
# n             : number of values in each array pred = [...] or real = [...]
# use_n_sub     : n_sub mode (True or False)
def readValidReport(fn, thresholdList, size, n, use_n_sub):

    # read validation report
    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # weight for each cell (left-top ~ right-bottom of output board)
    weight = [1]

    # read each line and compute MAE
    MAE = [] # MAE values (for each threshold)
    zeros = [] # count of number 0
    ones = [] # count of number 1
    weightedAvgResult = str(weight) + '\n' # to write valid_report_n_sub_X_weightedAvg.txt
    
    for i in range(len(thresholdList)):
        MAE.append(0.0)
        zeros.append(0.0)
        ones.append(0.0)
        
    lines = len(fl)-9 # number of lines
    
    for i in range(lines):

        # read predicted and real values
        pred = fl[i].split('pred = [')[1].split(']')[0] # pred = [...]
        real = fl[i].split('real = [')[1].split(']')[0] # real = [...]

        # convert to float
        predSplit = np.array(' '.join(pred.split()).split(' ')).astype('float')
        realSplit = np.array(' '.join(real.split()).split(' ')).astype('float')

        # check if length of predSplit and realSplit are the same
        try:
            assert(len(predSplit) == len(realSplit))
        except:
            print('ERROR: Length of prediction (pred = [...]) and real (real = [...]) are not the same.')
            return

        # <<< use n_sub mode >>>
        if use_n_sub == True:

            # center of output vector
            center = int((n-1)/2)
            
            # index of the line (indexed by i) for the board
            board_i = i % (size * size)
            board_i_y = int(board_i / size) # y-axis value of index i of the board
            board_i_x = board_i % size # x-axis value of index i of the board

            # find weighted prediction
            (weightedPred, refList, refVals, refWeights) = getWeightedPrediction(n, i, size, board_i_y, board_i_x, weight, True, fl)

            # divide by sum of the weights and use it as prediction
            weightedAvgResult += ('[' + str(i) + '] refList=' + str(refList) + ' refVals=' + str(refVals) + ' refWeights=' + str(refWeights)
                                  + ' weightedPred=' + str(round(weightedPred, 6)) + '\n')

            # compute MAE for each threshold
            for j in range(len(thresholdList)):
                
                # pred is considered as 1 -> MAE=0 if real=1, MAE=1 if real=-1
                if weightedPred > thresholdList[j]:
                    MAE[j] += (0.5 - int(realSplit[center])*0.5)
                    ones[j] += 1.0

                # pred is considered as 0 -> MAE=0 if real=-1, MAE=1 if real=1
                else:
                    MAE[j] += (int(realSplit[center])*0.5 + 0.5)
                    zeros[j] += 1.0

        # <<< use normal mode >>>
        else:

            # compute MAE for each threshold
            for j in range(len(thresholdList)):
                for k in range(len(predSplit)):
                
                    # pred is considered as 1 -> MAE=0 if real=1, MAE=1 if real=-1
                    if predSplit[k] > thresholdList[j]:
                        MAE[j] += (0.5 - int(realSplit[center])*0.5)
                        ones[j] += 1.0

                    # pred is considered as 0 -> MAE=0 if real=-1, MAE=1 if real=1
                    else:
                        MAE[j] += (int(realSplit[center])*0.5 + 0.5)
                        zeros[j] += 1.0

    # MAE <- MAE / (lines * n)
    # zeros,ones <- zeros,ones / (number of total count)
    for i in range(len(thresholdList)):

        # for MAE, divide by the number of rows
        MAE[i] /= lines

        # divide by length of output vector for normal mode
        if use_n_sub == False: MAE[i] /= len(predSplit)

        # for zeros and ones, divide to make the sum 1.0
        zeros_ = zeros[i]
        ones_ = ones[i]
        zeros[i] = zeros[i] / (zeros_ + ones_)
        ones[i] = ones[i] / (zeros_ + ones_)
        
    # print MAE
    result = '' # to write valid_report_n_sub_X_MAE.txt
    for i in range(len(thresholdList)):
        result_ = ('MAE when threshold =\t' + str(round(thresholdList[i], 6)) + '\tis\t' + str(round(MAE[i], 6))
                   + '\t0 in pred =\t' + str(round(zeros[i], 6)) + '\t1 in pred =\t' + str(round(ones[i], 6)))
        print(result_)
        result += result_ + '\n'

    # write to file
    f = open(fn[:len(fn)-4] + '_MAE.txt', 'w')
    f.write(result)
    f.close()

    # write weighted average only when use_n_sub is True
    if use_n_sub == True:
        f = open(fn[:len(fn)-4] + '_weightedAvg.txt', 'w')
        f.write(weightedAvgResult)
        f.close()

if __name__ == '__main__':
    thresholdList = [] # list of thresholds to use
    size = 25 # the number of rows/columns in each input data
    outputSize = 1 # the number of rows/columns in each output data
    outputLength = outputSize * outputSize # length of output vector
    
    for i in range(-20, 221): thresholdList.append(round(0.01*i - 1, 6))
    print(thresholdList)
    
    readValidReport('valid_report_n_sub_0.txt', thresholdList, size, outputLength, True)
    readValidReport('valid_report_n_sub_1.txt', thresholdList, size, outputLength, True)
    readValidReport('valid_report_n_sub_2.txt', thresholdList, size, outputLength, True)
    readValidReport('valid_report_n_sub_3.txt', thresholdList, size, outputLength, True)
    readValidReport('valid_report_n_sub_4.txt', thresholdList, size, outputLength, True)
