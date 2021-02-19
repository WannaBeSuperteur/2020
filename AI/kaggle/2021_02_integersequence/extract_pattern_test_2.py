import math
import copy
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.stats import mode
import random

import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # count
    count = 0
    correct = 0

    # read file
    f = open('test.csv', 'r')
    fl = f.readlines()
    fl = fl[1:]
    f.close()

    input_ = []
    output = []

    for i in range(len(fl)):
        if i % 2500 == 0: print(i)
        
        sequence = fl[i].split('"')[1].split(',')

        # 0 ~ last as input
        for j in range(len(sequence)):
            sequence[j] = int(sequence[j])

        # append to data
        input_.append(sequence)

    for i in range(len(input_)):
        if i % 1500 == 0: print(i)
            
        # find answer
        this = input_[i]

        try:
            prediction = mode(this)[0][0]
        except:
            prediction = 0

        ### prediction algorithm ###
        
        try:
        
            # difs
            difs = []
            for j in range(len(this)-1):
                difs.append(this[j+1] - this[j])

            # dif2s
            dif2s = []
            for j in range(len(difs)-1):
                dif2s.append(difs[j+1] - difs[j])

            # dif3s
            dif3s = []
            for j in range(len(dif2s)-1):
                dif3s.append(dif2s[j+1] - dif2s[j])

            # dif4s
            dif4s = []
            for j in range(len(dif3s)-1):
                dif4s.append(dif3s[j+1] - dif3s[j])

            # dif5s
            dif5s = []
            for j in range(len(dif4s)-1):
                dif5s.append(dif4s[j+1] - dif4s[j])

            # use dif5s ~ difs
            arrays = [dif5s, dif4s, dif3s, dif2s, difs]
            arrayNo = -1
            
            for j in range(len(arrays)):
                
                arr = arrays[j]
                if len(arr) < 5: continue

                # predict a(N+1) based on a(N-2) ~ a(N)
                last3 = [arr[-3], arr[-2], arr[-1]]
                predictions = []

                for k in range(len(arr)-3):
                    consec3 = [arr[k], arr[k+1], arr[k+2]]
                    if consec3 == last3: predictions.append(arr[k+3])

                # predict a(N+1) based on a(N-1) ~ a(N)
                if len(predictions) == 0:
                    last2 = [arr[-2], arr[-1]]

                    for k in range(len(arr)-2):
                        consec2 = [arr[k], arr[k+1]]
                        if consec2 == last2: predictions.append(arr[k+2])

                # predict a(N+1) based on only a(N)
                if j == 4 and len(predictions) == 0:
                    last1 = arr[-1]

                    for k in range(len(arr)-1):
                        consec1 = arr[k]
                        if consec1 == last1: predictions.append(arr[k+1])

                if len(predictions) > 0:
                    pre_prediction = mode(predictions)[0][0]
                    arrayNo = j

            # derive final prediction using pre_prediction
            if len(predictions) > 0:

                difsL = difs[-1]
                dif2sL = dif2s[-1]
                dif3sL = dif3s[-1]
                dif4sL = dif4s[-1]
                dif5sL = dif5s[-1]
            
                if arrayNo == 0:
                    dif4_prediction = dif4sL + pre_prediction
                    dif3_prediction = dif3sL + dif4_prediction
                    dif2_prediction = dif2sL + dif3_prediction
                    dif_prediction = difsL + dif2_prediction
                    prediction = this[-1] + dif_prediction

                elif arrayNo == 1:
                    dif3_prediction = dif3sL + pre_prediction
                    dif2_prediction = dif2sL + dif3_prediction
                    dif_prediction = difsL + dif2_prediction
                    prediction = this[-1] + dif_prediction
                
                elif arrayNo == 2:
                    dif2_prediction = dif2sL + pre_prediction
                    dif_prediction = difsL + dif2_prediction
                    prediction = this[-1] + dif_prediction

                elif arrayNo == 3:
                    dif_prediction = difsL + pre_prediction
                    prediction = this[-1] + dif_prediction

                elif arrayNo == 4:
                    prediction = this[-1] + pre_prediction

        except:
            pass
        
        output.append([int(round(prediction))])

    RD.saveArray('output.txt', output, '\t', 500)
