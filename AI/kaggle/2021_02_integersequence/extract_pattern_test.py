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
        if i % 250 == 0: print(i)
            
        # answer
        input_[i] = input_[i][5:]
        input_saved = copy.deepcopy(input_[i])
        
        errors = [] # ax^3+bx^2+cx+d error for each SL
        
        minError = 1000000
        minErrorSL = None
        minErrorGoalY = None

        triangleCases = int(math.sqrt(len(input_[i]) * len(input_[i]) / 2))

        for TC in range(triangleCases + 1): # triangle cases

            # use original array
            if TC == 0:
                input__ = copy.deepcopy(input_[i])

            # use array[0], array[1], array[3], array[6], array[10], ...
            else:
                input__ = []
                for tc in range(TC):
                    try:
                        input__.append(input_[i][TC + int(tc*(tc+1)/2)])
                    except:
                        pass

            for SL in range(1, 51): # season length

                # make difference of sub-sequence
                subSeq = []
                
                for j in range(len(input__)-SL, SL-1, -SL):
                    subSeq = [input__[j] - input__[j-SL]] + subSeq
                    
                # try optimize
                if len(subSeq) > 10:

                    try:
                        # optimize
                        sx = np.array(subSeq[:len(subSeq)-1]).astype(float)
                        sy = np.array(subSeq[1:]).astype(float)
                        sc = np.polyfit(sx, sy, 2)

                        sy_fit = sc[0] * pow(sx, 2) + sc[1] * sx + sc[2]

                        # find the next value considering SL
                        goal_dif = subSeq[len(subSeq)-1]
                        goal_y = input__[len(input__)-SL] + sc[0] * pow(goal_dif, 2) + sc[1] * goal_dif + sc[2]

                        # update error
                        error = abs(sy[-1] - sy_fit[-1])

                        if error < minError:
                            minError = error
                            minErrorSL = SL
                            minErrorGoalY = goal_y

                    except:
                        pass

        # use mode if minError is too large
        if minError > 0.5:
            #print(mode(input_saved)[0])
            
            try:
                minErrorGoalY = mode(input_saved[max(0, len(input_saved)-3):])[0][0]
            except:
                minErrorGoalY = 0

        #print(i, input_[i][len(input_[i])-5:], minErrorGoalY)
        output.append([int(round(minErrorGoalY))])

    RD.saveArray('output.txt', output, '\t', 500)
