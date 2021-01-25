import math
import numpy as np
import readData as RD

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

    ### read ID to validate, from the validation report
    id0ToValidate = []
    
    report = open(fn, 'r')
    rows = report.readlines()
    leng = len(rows)
    report.close()

    # using parsing
    for i in range(leng-9):
        id_ = rows[i].split(']')[0][1:]
        id0ToValidate.append(int(id_))

    print(np.array(id0ToValidate))

    ### add randomly select rows to validate, from delta = 2 to 5, using validRate
    # ID: delta 1 = 000000 ~ 624999
    #     delta 2 = 625000 ~ 1.249M
    #     delta 3 = 1.250M ~ 1.874M
    #     delta 4 = 1.875M ~ 2.499M
    #     delta 5 = 2.500M ~ 3.124M (total 3,125,000 rows)
    # for delta = 2 to delta = 5, use (line No.) + 625000 * (delta - 1)
    
    # T: training, V: validation, then, for example
    # training data, delta = 1 -> TTTTTTTTVV
    # training data, delta = 2 -> ...V..V...
    # training data, delta = 3 -> .V.V......
    # training data, delta = 4 -> ..V.....V.
    # training data, delta = 5 -> ......VV..

    # load id and training input file
    id0 = RD.loadArray('train_id_sub_0.txt')
    id1 = RD.loadArray('train_id_sub_1.txt')
    id2 = RD.loadArray('train_id_sub_2.txt')
    id3 = RD.loadArray('train_id_sub_3.txt')
    id4 = RD.loadArray('train_id_sub_4.txt')
    
    if use_n_sub == True:
        train0 = RD.loadArray('train_input_n_sub_0.txt')
        train1 = RD.loadArray('train_input_n_sub_1.txt')
        train2 = RD.loadArray('train_input_n_sub_2.txt')
        train3 = RD.loadArray('train_input_n_sub_3.txt')
        train4 = RD.loadArray('train_input_n_sub_4.txt')
    else:
        train0 = RD.loadArray('train_input_sub_0.txt')
        train1 = RD.loadArray('train_input_sub_1.txt')
        train2 = RD.loadArray('train_input_sub_2.txt')
        train3 = RD.loadArray('train_input_sub_3.txt')
        train4 = RD.loadArray('train_input_sub_4.txt')

    # list of training data and validating IDs
    trainData = [train0, train1, train2, train3, train4]
    validID = [id0, id1, id2, id3, id4]

    # extract data to validate from train0, using ID list id0ToValidate
    # FILL IN THE BLANK

    # randomly select rows for delta = 2 to 5
    # FILL IN THE BLANK

    ### validate (get output for validation input) using model of modelName
    # for delta = 1 to 5
    # FILL IN THE BLANK

    ### compute loss (binary)
    # FILL IN THE BLANK

    ### write validation report (name: fn = report.txt -> file name = report_repeatDelta.txt)
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
