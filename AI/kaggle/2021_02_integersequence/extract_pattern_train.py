import math
import copy
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.stats import mode
import random

if __name__ == '__main__':

    # count
    count = 0
    correct = 0

    # read file
    f = open('test.csv', 'r')
    fl = f.readlines()
    fl = fl[1:]
    f.close()

    temp = []
    for i in range(12):
        for j in range(i):
            temp.append(pow(2, j+1)-1 + i)
    print(temp)
    temp = temp[:len(temp)-8]
            
    input_ = [temp[:len(temp)-1]]
    output_ = [temp[1:]]

    for i in range(40000):
        if i % 2500 == 0: print(i)
        
        sequence_ = fl[i].split('"')[1].split(',')

        # 0 ~ (last) as input
        inputData = sequence_
        
        for i in range(len(inputData)):
            inputData[i] = int(inputData[i])

        # append to data
        input_.append(inputData)

    for i in range(len(input_)):
            
        # answer
        answer = input_[i][len(input_[i]) - 1]
        input_[i] = input_[i][:len(input_[i]) - 1]
        input_saved = copy.deepcopy(input_[i])
        
        errors = [] # ax^3+bx^2+cx+d error for each SL
        
        minError = 1000000
        minErrorSL = None
        minErrorGoalY = None

        triangleCases = int(math.sqrt(len(input_[i]) * len(input_[i]) / 2))
        print('answer:', answer)

        for TC in range(triangleCases + 1): # triangle cases

            # use original array
            if TC == 0:
                input__ = copy.deepcopy(input_[i])

            # use array[0], array[1], array[3], array[6], array[10], ...
            else:
                input__ = []
                for tc in range(triangleCases + 1):
                    try:
                        input__.append(input_[i][TC + int(tc*(tc+1)/2)])
                    except:
                        pass

            # min length of subseq
            if TC == 0:
                minLength = 10
            else:
                minLength = 4

            print(TC, input__)

            for SL in range(1, 51): # season length

                # make difference of sub-sequence
                subSeq = []
                
                for j in range(len(input__)-SL, SL-1, -SL):
                    subSeq = [input__[j] - input__[j-SL]] + subSeq

                # try optimize
                if len(subSeq) >= minLength + 3:

                    try:
                        # optimize
                        sx = np.array(subSeq[3:len(subSeq)-1]).astype(float)
                        sy = np.array(subSeq[4:]).astype(float)
                        sc = np.polyfit(sx, sy, 2)

                        sy_fit = sc[0] * pow(sx, 2) + sc[1] * sx + sc[2]

                        # find the next value considering SL
                        goal_dif = subSeq[len(subSeq)-1]

                        if TC == 0:
                            goal_y = input_[i][len(input_[i])-SL] + sc[0] * pow(goal_dif, 2) + sc[1] * goal_dif + sc[2]
                        else:
                            goal_y = input__[len(input__)-1] + sc[0] * pow(goal_dif, 2) + sc[1] * goal_dif + sc[2]

                        # update error
                        error = abs(sy[-1] - sy_fit[-1])

                        if error < minError:
                            minError = error
                            minErrorSL = SL
                            minErrorGoalY = goal_y

                    except:
                        pass

            print('minError:' , minError)
            print('minErrorSL:', minErrorSL)
            print('minErrorGoalY:', minErrorGoalY)

        # use mode if minError is too large
        if minError > 0.5:
            #print(mode(input_saved)[0])

            try:
                minErrorGoalY = mode(input_saved[max(0, len(input_saved)-3):])[0][0]
            except:
                minErrorGoalY = 0
        
        # error
        count += 1

        print(answer, minErrorGoalY)

        try:
            ERR = abs(round(answer - minErrorGoalY, 6))
            if ERR < 0.5: correct += 1
        except:
            ERR = 'error'

        # print error
        #try:
        #    print('error (i=' + str(i) + '): ' + str(ERR))
        #except:
        #    print('error (i=' + str(i) + '): error')

        print(str(correct) + ' / ' + str(count))
