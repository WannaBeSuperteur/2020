import math

# Naive Bayes model (return Y value, which is True or False)
# xs         : array contains value of Xs - form of [x0s, x1s, ...] where x0s = [x0_0, x0_1, ...], x1s = [x1_0, x1_1, ...], ...
# ys         : array contains value of Ys - form of [y_0, y_1, ...] where each value is True or False
# xConditions: predict if 'y' or 'not y' using this x condition - form of [v0, v1, ...] where x0=v0, x1=v1, ...
def naiveBayes(xs, ys, xConditions):

    countY = 0 # number of value 'True' in ys

    probForXgivenY = [0]*len(xs) # P(xi=vi|y) = P(x0=v0|y), P(x1=v1|y), ..., P(xk_vk|y)
    probForXgivenNotY = [0]*len(xs) # P(xi=vi|~y) = P(x0=v0|~y), P(x1=v1|~y), ..., P(xk_vk|~y)

    # calculate countY and probForY
    for j in range(len(ys)):
        if ys[j] == True: countY += 1
    probForY = countY / len(ys) # P(y)
    probForNotY = 1 - probForY # P(~y)

    # calculate P(xi|y) and P(xi|~y) for each i
    for i in range(len(xs)):
        countX = 0 # number of value xConditions[i]=xi (xi=vi) in xs[i]

        # calculate probForXgivenY and probForXgivenNotY
        for j in range(len(xs[i])):
            if xConditions[i] == xs[i][j]: # if xi[j]=v[j]
                countX += 1

                # add 1 to probForXgivenY if y is true, probForXgivenNotY otherwise
                if ys[j] == True: probForXgivenY[i] += 1 # xi[j]=v[j] and y
                else: probForXgivenNotY[i] += 1 # xi[j]=v[j] and ~y

        # P(xi=vi|y) = P(xi=vi and y)/P(y)
        # P(xi=vi|~y) = P(xi=vi and ~y)/P(~y)
        probForXgivenY[i] /= countY
        probForXgivenNotY[i] /= (len(ys) - countY)

    # calculate P(y)P(x1|y)P(x2|y)...P(xn|y)
    probForYgivenXs = probForY
    for i in range(len(xs)): probForYgivenXs *= probForXgivenY[i]

    # calculate P(~y)P(x1|~y)P(x2|~y)...P(xn|~y)
    probForNotYgivenXs = probForNotY
    for i in range(len(xs)): probForNotYgivenXs *= probForXgivenNotY[i]

    print('P(y)              = ' + str(round(probForY, 6)))
    for i in range(len(probForXgivenY)): print('P(xi=vi | y)  [' + str(i) + '] = ' + str(round(probForXgivenY[i], 6)))
    print('Final for y=True  = ' + str(round(probForYgivenXs*100, 6)) + '%')
    print('')
    print('P(~y)             = ' + str(round(probForNotY, 6)))
    for i in range(len(probForXgivenNotY)): print('P(xi=vi | ~y) [' + str(i) + '] = ' + str(round(probForXgivenNotY[i], 6)))
    print('Final for y=False = ' + str(round(probForNotYgivenXs*100, 6)) + '%')

    # return
    print('')
    if probForYgivenXs >= probForNotYgivenXs: return True
    else: return False

if __name__ == '__main__':

    # https://www.slideshare.net/MohammedAbdallaYouss/machine-learning-clisification-algorthims
    print('----\n\n <<< Naive Bayes Test 1 >>>\n')
    nvTest = naiveBayes([['s', 'o', 'r', 's', 's', 'o', 'r', 'r', 's', 'r', 's', 'o', 'o', 'r']],
                        [False, True, True, True, True, True, False, False, True, True, False, True, True, False], ['r'])
    print('Naive Bayes test = ' + str(nvTest))

    # https://www.saedsayad.com/naive_bayesian.htm
    print('\n----\n\n <<< Naive Bayes Test 2 >>>\n')
    nvTest = naiveBayes([['r', 'r', 'o', 's', 's', 's', 'o', 'r', 'r', 's', 'r', 'o', 'o', 's'],
                         ['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'],
                         ['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'],
                         ['f', 't', 'f', 'f', 'f', 't', 't', 'f', 'f', 'f', 't', 't', 'f', 't']],
                        [False, False, True, True, True, False, True, False, True, True, True, True, True, False],
                        ['r', 'c', 'h', 't'])
    print('Naive Bayes test = ' + str(nvTest))

    # https://medium.com/datadriveninvestor/naive-bayes-d36e57d80652
    print('\n----\n\n <<< Naive Bayes Test 3 >>>\n')
    nvTest = naiveBayes([[1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                         ['n', 'n', 'y', 'y', 's', 'n', 'n', 'n', 'n', 'y', 's', 's', 'n', 'y', 's', 'n', 'n', 'n', 'y', 'n'],
                         [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]],
                        [True, False, False, True, False, False, True, False, True, False, False, False, True, False, False, False, True, False, True, False],
                        [1, 0, 'y', 0])
    print('Naive Bayes test = ' + str(nvTest))
