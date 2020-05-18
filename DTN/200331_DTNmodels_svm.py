import math
import random
import numpy as np
from matplotlib import pyplot as plt

# find and return the most accurate hyperplane
# criteria 1: more accurate / criteria 2: larger margin

# hyperplanes    : info about hyperplanes (ax+by+c=0) (form: [[a, b, c, margin, accuracyRate, plusClass], ...])
# -> margin      : margin of hyperplane (distance between correctly classified point nearest to boundary and hyperplane)
# -> accuracyRate: (# of correctly classified points) / (# of all points)
# -> plusClass   : N -> point (x, y) that ax+by+c>0 is classified as class N
def mostAccurateHyperplane(hyperplanes):

    # remove None
    for i in range(len(hyperplanes)-1, -1, -1):
        if hyperplanes[i] == None: hyperplanes.pop(i)

    # sort by (1) more accurate (2) larger margin 
    sortedHyerplanes = list(reversed(sorted(hyperplanes, key=lambda x: (x[4], x[3]))))
    return sortedHyerplanes[0]

# make length-1 dataset from data (in the form of [[data0], [data1], ...])
def makeLength1Dataset(data):
    result = []
    for i in range(len(data)): result.append([data[i]])
    return result

# make length-2 dataset from data (in the form of [[data0, data1], [data0, data2], ...])
def makeLength2Dataset(data):
    result = []
    for i in range(len(data)):
        for j in range(i): result.append([data[j], data[i]])
    return result

# decide hyperplane for each 'order'
# data   : dataset (in the form of [[x0, y0], [x1, y1], ..., [xn, yn]])
# classes: class of data (in the form of [c0, c1, c2, ...])
# order  : 0 -> for every [data0, data1] in class 0 and [data2] in class 1
#        : 1 -> for every [data0] in class 0 and [data1, data2] in class 1
def decideHyperplane(data, classes, order):

    # append each data to class0data if class is 0, class1data if class is 1
    class0data = []
    class1data = []
    for i in range(len(data)):
        if classes[i] == 0: class0data.append(data[i]) # class 0
        else: class1data.append(data[i]) # class 1

    # make length-1 and length-2 dataset
    length1 = [] # length-1 dataset - in the form of [[data0], [data1], ...]
    length2 = [] # length-2 dataset - in the form of [[data0, data1], [data0, data2], ...]
    
    if order == 0: # [data0, data1](0) [data2](1)
        length1 = makeLength1Dataset(class1data)
        length2 = makeLength2Dataset(class0data)
    elif order == 1: # [data0](0) [data1, data2](1)
        length1 = makeLength1Dataset(class0data)
        length2 = makeLength2Dataset(class1data)

    # for each pair (length1[i], length2[j]), decide the best hyperplane
    hyperplanes = []
    for i in range(len(length1)):
        print('point ' + str(i))
        for j in range(len(length2)):

            # suppose that for each pair ([data0], [data1, data2])
            dataInLength1 = length1[i][0] # data0
            dataInLength2_0 = length2[j][0] # data1
            dataInLength2_1 = length2[j][1] # data2

            ###print('\n\nL1: ' + str(length1[i]) + ' L2: ' + str(length2[j]))

            L1X = dataInLength1[0] # x axis of data0
            L1Y = dataInLength1[1] # y axis of data0
            L2X0 = dataInLength2_0[0] # x axis of data1
            L2Y0 = dataInLength2_0[1] # y axis of data1
            L2X1 = dataInLength2_1[0] # x axis of data2
            L2Y1 = dataInLength2_1[1] # y axis of data2

            # find hyperplane using the data
            # 0. find vector v from L2_0 to L2_1 (the hyperplane is parallel to the vector)
            v = [L2X1-L2X0, L2Y1-L2Y0]
            ###print('v: ' + str(v))
            
            # 1. find distance d between L1 and the line connecting L2_0 and L2_1
            # line ax+by+c: y - L2_0Y                               = (L2_1Y - L2_0Y)/(L2_1X - L2_0X) * (x - L2_0X)
            #               (L2_1X - L2_0X)y - L2_0Y(L2_1X - L2_0X) = (L2_1Y - L2_0Y)x - L2_0X(L2_1Y - L2_0Y)
            #               (L2_1Y - L2_0Y)x + (L2_0X - L2_1X)y + L2_0Y(L2_1X - L2_0X) - L2_0X(L2_1Y - L2_0Y) = 0
            #               a = L2_1Y - L2_0Y, b = L2_0X - L2_1X, c = L2_0Y(L2_1X - L2_0X) - L2_0X(L2_1Y - L2_0Y)
            # distance    : |ax+by+c|/sqrt(a^2+b^2) where (x, y) = (L1X, L1Y)
            a = L2Y1-L2Y0
            b = L2X0-L2X1
            c = L2Y0*(L2X1-L2X0)-L2X0*(L2Y1-L2Y0)
            ###print('line connection L20 and L21: ' + str(a) + 'x + ' + str(b) + 'y + ' + str(c) + ' = 0')

            if a == 0 and b == 0:
                print(str(j) + ' -> divide by 0')
                continue # to prevent div by 0 error
            
            dist = abs(a * L1X + b * L1Y + c) / math.sqrt(a*a + b*b)
            ###print('dist: ' + str(dist))
            
            # 2. the hyperplane passes point m, the destination point, from L1 toward vector vOrt (orthogonal with v) with distance d/2
            # <a, b> and <-b, a> are orthogonal
            vOrt = [L2Y0-L2Y1, L2X1-L2X0]
            ###print('vOrt: ' + str(vOrt))

            if L2Y0-L2Y1 == 0 and L2X0-L2X1 == 0:
                print(str(j) + ' -> divide by 0')
                continue # to prevent div by 0 error

            # to make |vOrt| = dist, vOrt = <-b, a> -> vOrt = dist * <-b/sqrt(a^2+b^2), a/sqrt(a^2+b^2)>
            vOrt[0] *= (dist / math.sqrt(pow(L2Y0-L2Y1, 2) + pow(L2X1-L2X0, 2))) # x of vOrt
            vOrt[1] *= (dist / math.sqrt(pow(L2Y0-L2Y1, 2) + pow(L2X1-L2X0, 2))) # y of vOrt
            m = [L1X + vOrt[0] / 2, L1Y + vOrt[1] / 2]
            ###print('vOrt: ' + str(vOrt))
            ###print('m: ' + str(m))
            
            # 3. calculate formula for the hyperplane using vOrt and m
            # vector vOrt=<vOrt_x, vOrt_y>, point m=(mx, my) -> vOrt_x(x-mx)+vOrt_y(y-my)=0 -> (vOrt_x)x + (vOrt_y)y - (vOrt_x*mx + vOrt_y*my) = 0
            hpA = vOrt[0] # vOrt_x
            hpB = vOrt[1] # vOrt_y
            hpC = -(vOrt[0]*m[0]+vOrt[1]*m[1]) # -(vOrt_x*mx + vOrt_y*my)
            ###print('hyperplane eq: ' + str(hpA) + 'x + ' + str(hpB) + 'y + ' + str(hpC) + ' = 0')

            # 4. compute accuracy rate
            estimatedClasses = [] # estimate class for each data in array 'data'
            for k in range(len(data)):
                if hpA * data[k][0] + hpB * data[k][1] + hpC > 0: estimatedClasses.append(1)
                else: estimatedClasses.append(0)

            # compareList[0]: number of cases, both actual class and estimated class are 0
            # compareList[1]: number of cases, actual class is 0 but estimated class is 1
            # compareList[2]: number of cases, actual class is 1 but estimated class is 0
            # compareList[3]: number of cases, both actual class and estimated class are 1
            compareList = [0, 0, 0, 0]
            for k in range(len(data)):
                if classes[k] == 0 and estimatedClasses[k] == 0: compareList[0] += 1
                elif classes[k] == 0 and estimatedClasses[k] == 1: compareList[1] += 1
                elif classes[k] == 1 and estimatedClasses[k] == 0: compareList[2] += 1
                elif classes[k] == 1 and estimatedClasses[k] == 1: compareList[3] += 1

            accuracyNotReverseEstimatedClass = compareList[0] + compareList[3] # accuracy when not reversing(0 <-> 1) estimated classes
            accuracyReverseEstimatedClass = compareList[1] + compareList[2] # accuracy when reversing(0 <-> 1) estimated classes
            accuracyRate = max(accuracyNotReverseEstimatedClass, accuracyReverseEstimatedClass)/len(data)

            # plusClass: N -> point (x, y) that ax+by+c>0 is classified as class N
            if accuracyNotReverseEstimatedClass > accuracyReverseEstimatedClass: plusClass = 1
            else: plusClass = 0

            if hpA == 0 and hpB == 0:
                print(str(j) + ' -> divide by 0')
                continue # to prevent div by 0 error

            # append to list 'hyperplanes' -> form: [a, b, c, margin, accuracyRate, plusClass] where hyperplane : ax+by+c=0
            # margin: FIND SMALLEST MARGIN BETWEEN CORRECTLY CLASSIFIED POINTS
            smallestMargin = 10000 # distance between correctly classified point nearest to boundary and hyperplane, (ax+by+c)/sqrt(a^2+b^2)
            for k in range(len(data)):

                # don't count for incorrectly classified points
                if classes[k] == estimatedClasses[k] and plusClass == 0: continue
                if classes[k] != estimatedClasses[k] and plusClass == 1: continue
                
                thisMargin = abs(hpA * data[k][0] + hpB * data[k][1] + hpC) / math.sqrt(hpA*hpA + hpB*hpB)
                if thisMargin < smallestMargin: smallestMargin = thisMargin

            ###print('smallest margin: ' + str(smallestMargin))

            # append to list
            hyperplanes.append([hpA, hpB, hpC, smallestMargin, accuracyRate, plusClass])

    # find and return the most accurate hyperplane in 'hyperplanes'
    return mostAccurateHyperplane(hyperplanes)

# find max and min value of x (in both 'data' and 'newData')
def findRangeOfX(data, newData):
    minX = 10000
    maxX = -10000
    
    for i in range(len(data)):
        if minX > data[i][0]: minX = data[i][0]
        if maxX < data[i][0]: maxX = data[i][0]
    for i in range(len(newData)):
        if minX > newData[i][0]: minX = newData[i][0]
        if maxX < newData[i][0]: maxX = newData[i][0]

    return (minX-0.5, maxX+0.5)

# Support Vector Machine
# data        : dataset (in the form of [[x0, y0], [x1, y1], ..., [xn, yn]])
# classes     : class of data (in the form of [c0, c1, c2, ...])
# newData     : new data to classify (in the form of [[x0, y0, class0], [x1, y1, class1], ..., [xn, yn, classn]]
# displayChart: whether or not display graph for data
def SVM(data, classes, newData, displayChart):
    assert(len(data) == len(classes)) # number of data = number of classes
    assert(len(data[0]) == 2) # check in the form of [x0, y0]

    # for every [data0, data1] in class 0 and [data2] in class 1, decide SV
    hyperplaneForOrder0 = None
    if sum(classes) > 0 and sum(classes) < len(classes)-1:
        hyperplaneForOrder0 = decideHyperplane(data, classes, 0)

    # for every [data0] in class 0 and [data1, data2] in class 1, decide SV
    hyperplaneForOrder1 = None
    if sum(classes) > 1 and sum(classes) < len(classes):
        hyperplaneForOrder1 = decideHyperplane(data, classes, 1)

    # find better hyperplane between hyperplaneForOrder0 and hyperplaneForOrder1
    # print(hyperplaneForOrder0, hyperplaneForOrder1)
    if hyperplaneForOrder0 == None and hyperplaneForOrder1 == None:
        print("All data was classified in the same class -> can't generate hyperplane")
        return
    bestHyperplane = mostAccurateHyperplane([hyperplaneForOrder0, hyperplaneForOrder1])

    a = bestHyperplane[0]
    b = bestHyperplane[1]
    c = bestHyperplane[2]
    plusClass = bestHyperplane[5] # N -> point (x, y) that ax+by+c>0 is classified as class N

    # display best hyperplane ax+by+c = 0 -> by = -ax-c, y = (-a/b)x-c/b
    (minX, maxX) = findRangeOfX(data, newData)
    xh = np.linspace(minX, maxX, 1000)
    yh = (-a/b)*xh + (-c/b)
    if displayChart: plt.plot(xh, yh, '-g', label='best hyperplane by SVM')

    # print (0, 0)
    if displayChart: plt.plot(0, 0, 'k.')

    ###print(bestHyperplane)

    # display graph
    if displayChart:
        for i in range(len(data)):
            if classes[i] == 0: plt.plot(data[i][0], data[i][1], 'bs')
            else: plt.plot(data[i][0], data[i][1], 'rs')

    # classify new data using 'bestHyperplane' (ax+by+c=0)
    # form of each hyperplane is [a, b, c, margin, accuracyRate, plusClass]
    correct = 0

    legend = [0, 0, 0, 0] # TP, TN, FP, FN

    for i in range(len(newData)):
        thisNewData = newData[i] # in the form of [xi, yi]
        x = thisNewData[0]
        y = thisNewData[1]

        # class prediction for newData[i]
        if a*x + b*y + c > 0: predictedClass = plusClass
        else: predictedClass = 1-plusClass

        if displayChart: # thisNewData[2] : actual class
            if predictedClass == 0 and thisNewData[2] == 0: # True Negative
                legend[1] += 1
                plt.plot(x, y, 'b.')
            elif predictedClass == 1 and thisNewData[2] == 0: # False Positive
                legend[2] += 1
                plt.plot(x, y, 'rx')
            elif predictedClass == 0 and thisNewData[2] == 1: # False Negative
                legend[3] += 1
                plt.plot(x, y, 'bx')
            elif predictedClass == 1 and thisNewData[2] == 1: # True Positive
                legend[0] += 1
                plt.plot(x, y, 'r.')

        # check if prediction is correct
        if predictedClass == thisNewData[2]: correct += 1

    print('True Positive  (red  O): ' + str(round(100 * legend[0] / len(newData), 2)) + '%')
    print('True Negative  (blue O): ' + str(round(100 * legend[1] / len(newData), 2)) + '%')
    print('False Positive (red  X): ' + str(round(100 * legend[2] / len(newData), 2)) + '%')
    print('False Negative (blue X): ' + str(round(100 * legend[3] / len(newData), 2)) + '%')

    print('Real Positive  (rO, bX): ' + str(round(100 * (legend[0] + legend[3]) / len(newData), 2)) + '%')
    print('Real Negative  (bO, rX): ' + str(round(100 * (legend[1] + legend[2]) / len(newData), 2)) + '%')

    print('correct rate   (rO, bO): ' + str(round(100 * correct / len(newData), 2)) + '%')
    if displayChart:
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    data = []
    classes = []
    newData = []
    
    theta = (random.random() * math.pi - math.pi / 2) / 2
    updown = random.random()

    option = int(input('0:linear, other:quadratic'))
    toTrain = int(input('number of training data'))
    toTest = int(input('number of test data'))

    # training data
    for i in range(toTrain):
        x = random.random() * 10 - 5
        y = random.random() * 10 - 5
        data.append([x, y])

        if option == 0: # linear
            if updown > 0.5:
                if y >= np.tan(theta)*x: classes.append(1)
                else: classes.append(0)
            else:
                if y >= np.tan(theta)*x: classes.append(0)
                else: classes.append(1)
        else: # quadratic
            if updown > 0.5:
                if y >= x*x*(updown-0.75): classes.append(1)
                else: classes.append(0)
            else:
                if y >= x*x*(updown-0.25): classes.append(0)
                else: classes.append(1)

    # test data
    for i in range(toTest):
        x = random.random() * 10 - 5
        y = random.random() * 10 - 5

        if option == 0: # linear
            if updown > 0.5:
                if y >= np.tan(theta)*x: newDataClass = 1
                else: newDataClass = 0
            else:
                if y >= np.tan(theta)*x: newDataClass = 0
                else: newDataClass = 1
        else: # quadratic
            if updown > 0.5:
                if y >= x*x*(updown-0.75): newDataClass = 1
                else: newDataClass = 0
            else:
                if y >= x*x*(updown-0.25): newDataClass = 0
                else: newDataClass = 1
        
        newData.append([x, y, newDataClass])

    # display actual class-divider
    (minX, maxX) = findRangeOfX(data, newData)

    xa = np.linspace(minX, maxX, 1000)
    
    if option == 0: # linear
        print('tangent: ' + str(np.tan(theta)))
        ya = np.tan(theta)*xa
    elif updown > 0.5: # quadratic
        print('value of a: ' + str(updown-0.75))
        ya = xa*xa*(updown-0.75)
    else: # quadratic
        print('value of a: ' + str(updown-0.25))
        ya = xa*xa*(updown-0.25)
    
    plt.plot(xa, ya, '-k', label='actual divider')
        
    SVM(data, classes, newData, True)
