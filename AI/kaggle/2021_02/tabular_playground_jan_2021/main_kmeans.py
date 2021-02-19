import sys
sys.path.insert(0, '../../AI_BASE')

import numpy as np
import readData as RD
import deepLearning_main as DL
import kNN as AIBASE_KNN
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# merge training input and training output
def mergeTrain(TRI, TRO, TRIO):

    # read array
    TI = np.array(RD.loadArray(TRI, '\t'))
    TO = np.array(RD.loadArray(TRO, '\t'))

    # concatenate arrays
    TIO = np.concatenate((TI, TO), axis=1)

    # write array
    RD.saveArray(TRIO, TIO, '\t', 500)

# main
if __name__ == '__main__':

    # training input and test input:
    # NORMALIZED using avg and stddev of each training input column
    #
    # training output:
    # (original output value - 8.0)
    #
    # test output:
    # (converted output value + 8.0)

    # meta info
    TRI = 'train_input.txt'
    TRO = 'train_output.txt'
    TRIO = 'train_IO.txt'
    TEI = 'test_input.txt'

    # merge train input and output
    try:
        _ = open(TRIO, 'r')
        _.close()
    except:
        mergeTrain(TRI, TRO, TRIO)

    # K-means clustering
    finalResult = None
    trainName = 'train_IO.txt'
    testName = 'test_input.txt'
    ftype = 'txt'

    TRIO_array = RD.loadArray(TRIO, '\t')
    TEI_array = RD.loadArray(TEI, '\t')
    
    dfTrain = pd.DataFrame(TRIO_array)
    dfTest = pd.DataFrame(TEI_array)
    
    dfTestWeight = None
    caseWeight = False
    targetCol = 14
    targetIndex = 14

    k = 100
    useAverage = True

    # execute algorithm
    AIBASE_KNN.kNN(dfTrain, dfTest, dfTestWeight, caseWeight,
                   targetCol, targetIndex, k, useAverage)
