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
# fn            : file name of validation report
# thresholdList : list for thresholds (above->1, below->0)
# size          : input board size (width or height = 25)
# n             : number of values in each array pred = [...] or real = [...]
# modelName     : model name for validation
# validRate     : validation rate
# use_n_sub     : n_sub mode (True or False)
def valid(fn, thresholdList, size, n, modelName, validRate, use_n_sub):

    # read ID to validate, from the validation report
    IDtoValidate = []
    # FILL IN THE BLANK

    # add randomly select rows to validate, from delta = 2 to 5, using validRate
    # FILL IN THE BLANK
    
    # T: training, V: validation, then, for example
    # delta = 1 -> TTTTTTTTVV
    # delta = 2 -> ...V..V...
    # delta = 3 -> .V.V......
    # delta = 4 -> ..V.....V.
    # delta = 5 -> ......VV..

    # FILL IN THE BLANK

    # read model
    # FILL IN THE BLANK

    # validate (get output for validation input)
    # for delta = 1 to 5
    # FILL IN THE BLANK

    # compute loss (binary)
    # FILL IN THE BLANK

    # write validation report (name: fn = report.txt -> file name = report_repeatDelta.txt)
    # FILL IN THE BLANK

if __name__ == '__main__':
    thresholdList = [] # list of thresholds to use
    size = 25 # the number of rows/columns in each input data
    outputSize = 1 # the number of rows/columns in each output data
    outputLength = outputSize * outputSize # length of output vector
    
    for i in range(-20, 221): thresholdList.append(round(0.005*i, 6))
    readValidReport('valid_report_n_sub_0.txt', thresholdList, size, outputLength)
    readValidReport('valid_report_n_sub_1.txt', thresholdList, size, outputLength)
    readValidReport('valid_report_n_sub_2.txt', thresholdList, size, outputLength)
    readValidReport('valid_report_n_sub_3.txt', thresholdList, size, outputLength)
    readValidReport('valid_report_n_sub_4.txt', thresholdList, size, outputLength)
