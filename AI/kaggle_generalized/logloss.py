## form of submission file (*.csv):
# val00,val01,val02,...,val0n
# val10,val11,val12,...,val1n
# ...
# valk0,valk1,valk2,...,valkn
# ...
# valm0,valm1,valm2,...,valmn
# * each value is 0 or 1

## form of ground-truth file (*.txt)
# 111011011101111000010101001011111 (data with value 0 or 1)
# * let the length of above L+1

## compare: val0k with 0-th value of ground-truth file,
#           val1k with 1st  value of ground-truth file,
#           val2k with 2nd  value of ground-truth file,
#           val3k with 3rd  value of ground-truth file,
#           val4k with 4th  value of ground-truth file,
#           ..., and
#           valmin(m,L)k with min(m,L)-th value of ground-truth file
# if you input colNo as k

import math

# average log loss
def logLoss(correct, wrong, prob):
    correctLoss = correct * math.log(prob) * (-1)
    wrongLoss = wrong * math.log(1.0 - prob) * (-1)
    return (correctLoss + wrongLoss) / (correct + wrong)

if __name__ == '__main__':

    # read file info
    submission = input('submission file name')
    colNo = input('column of data from the submission file') # to get val01,val11, ..., you should input 1
    groundtruth = input('ground-truth file name')

    # arrays
    subArray = []
    GTArray = []

    # read submision file
    a = open(submission)
    aLines = a.readlines()
    a.close()

    for i in range(len(aLines)):
        aSplit = aLines[i].split('\n')[0].split(',')
        subArray.append(aSplit[int(colNo)])

    # read ground truth file
    b = open(groundtruth)
    bData = b.readlines()[0].split('\n')[0]
    b.close()

    for i in range(len(bData)): GTArray.append(bData[i])

    # print result
    correct = 0
    wrong = 0

    print('subm\tGT')
    length = min(len(aLines), len(bData))
    for i in range(length):
        if subArray[i] == GTArray[i]: correct += 1
        else: wrong += 1
        
        if i < min(20, length): print(str(subArray[i]) + '\t' + str(GTArray[i]))
    if length > 20: print('...')

    # print log loss
    print('correct: ' + str(correct) + ', wrong: ' + str(wrong))
    for i in range(50, 100):
        prob = i/100
        print('logloss val for prob = \t' + str(round(prob, 4)) + '\t : ' + str(logLoss(correct, wrong, prob)))
    for prob in [0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999]:
        print('logloss val for prob = \t' + str(round(prob, 4)) + '\t : ' + str(logLoss(correct, wrong, prob)))
