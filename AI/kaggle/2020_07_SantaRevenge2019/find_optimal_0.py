import helper as h
import random

FAMILIES = 6000

# main
if __name__ == '__main__':
    
    # read family data
    familyData = open('family_data.csv', 'r')
    data0 = familyData.readlines()
    familyData.close()

    famData = h.makeTable(data0, 1)

    # read sample submission data
    sampleSub = open('sample_submission.csv', 'r')
    data1 = sampleSub.readlines()
    sampleSub.close()

    subData = h.makeTable(data1, 1)
    lastScore = 2147483647

    # initialize each row of submission data as choice_0
    for i in range(FAMILIES):
        subData[i][1] = famData[i][1]

    ocA = h.getOccupancy(famData, subData)
    print(h.prefCost(famData, subData, ocA))
    print(h.accountPenalty(famData, subData, ocA))
    lastScore = h.getScoreFeasible(famData, subData, ocA)
    print(lastScore)
    print(ocA)

    # make feasible
    print('times 1')
    while True:
        ocA = h.getOccupancy(famData, subData)
        
        maxVal = -1
        minVal = 9999
        maxIndex = -1
        minIndex = -1
        for i in range(1, len(ocA)):
            if ocA[i] > maxVal:
                maxIndex = i
                maxVal = ocA[i]
        for i in range(1, len(ocA)):
            if ocA[i] < minVal:
                minIndex = i
                minVal = ocA[i]

        for i in range(len(subData)):
            if subData[i][1] == maxIndex:
                option = -1

                # subData <- famData[2]
                if subData[i][1] == famData[i][1] and famData[i][2] != maxIndex:
                    option = 0
                    subData[i][1] = famData[i][2] # subData <- famData[2]
                    
                elif subData[i][1] == famData[i][2]: option = 1
                elif subData[i][1] == famData[i][3]: option = 2
                elif subData[i][1] == famData[i][4]: option = 3
                elif subData[i][1] == famData[i][5]: option = 4
                elif subData[i][1] == famData[i][6]: option = 5
                elif subData[i][1] == famData[i][7]: option = 6
                elif subData[i][1] == famData[i][8]: option = 7
                elif subData[i][1] == famData[i][9]: option = 8

                # subData <- famData[1~9] or minIndex
                elif subData[i][1] == famData[i][10]:
                    option = 9
                    
                    randomBox = []
                    if famData[i][1] != maxIndex: randomBox.append(1)
                    if famData[i][2] != maxIndex: randomBox.append(2)
                    if famData[i][3] != maxIndex: randomBox.append(3)
                    if famData[i][4] != maxIndex: randomBox.append(4)
                    if famData[i][5] != maxIndex: randomBox.append(5)
                    if famData[i][6] != maxIndex: randomBox.append(6)
                    if famData[i][7] != maxIndex: randomBox.append(7)
                    if famData[i][8] != maxIndex: randomBox.append(8)
                    if famData[i][9] != maxIndex: randomBox.append(9)
                    randomBox.append(10)
                    
                    rand = random.randint(0, len(randomBox)-1)
                    if randomBox[rand] <= 9: subData[i][1] = famData[i][randomBox[rand]] # subData <- famData[1~9]
                    else: subData[i][1] = minIndex # subData <- minIndex

                # subData <- famData[1~option or option+2]
                if option >= 1 and option <= 8:
                    randomBox = []
                    for j in range(option):
                        if famData[i][j+1] != maxIndex: randomBox.append(j+1)
                    if famData[i][option+2] != maxIndex: randomBox.append(option+2)
                        
                    rand = random.randint(0, len(randomBox)-1)
                    subData[i][1] = famData[i][randomBox[rand]] # subData <- famData[1~option or option+2]
                    
                # print(option)
                break

        prefCost = h.prefCost(famData, subData, ocA)
        lastScore = h.getScoreFeasible(famData, subData, ocA)
        if prefCost > 1000000: break

        print('val: ' + str(maxVal) + ' ' + str(minVal) + ' / index: ' + str(maxIndex) + ' ' + str(minIndex) +
              ' / cost: ' + str(prefCost) + ' score: ' + str(lastScore))

    # make feasible
    print('times 2')
    while True:
        
        for i in range(len(subData)):
            subCopy = h.arrayCopy(subData)
            ocA = h.getOccupancy(famData, subCopy)
            
            option = -1
            if subCopy[i][1] == famData[i][9] and famData[i][8] != maxIndex:
                option = 0
                subCopy[i][1] = famData[i][8]
            elif subCopy[i][1] == famData[i][8] and famData[i][7] != maxIndex:
                option = 1
                subCopy[i][1] = famData[i][7]
            elif subCopy[i][1] == famData[i][7] and famData[i][6] != maxIndex:
                option = 2
                subCopy[i][1] = famData[i][6]
            elif subCopy[i][1] == famData[i][6] and famData[i][5] != maxIndex:
                option = 3
                subCopy[i][1] = famData[i][5]
            elif subCopy[i][1] == famData[i][5] and famData[i][4] != maxIndex:
                option = 4
                subCopy[i][1] = famData[i][4]
            elif subCopy[i][1] == famData[i][4] and famData[i][3] != maxIndex:
                option = 5
                subCopy[i][1] = famData[i][3]
            elif subCopy[i][1] == famData[i][3] and famData[i][2] != maxIndex:
                option = 6
                subCopy[i][1] = famData[i][2]
            elif subCopy[i][1] == famData[i][2] and famData[i][1] != maxIndex:
                option = 7
                subCopy[i][1] = famData[i][1]

            score = h.getScoreFeasible(famData, subCopy, ocA)
            if score < lastScore:
                lastScore = score
                subData = subCopy
                break

        print('val: ' + str(maxVal) + ' ' + str(minVal) + ' / index: ' + str(maxIndex) + ' ' + str(minIndex) +
              ' / cost: ' + str(prefCost) + ' score: ' + str(h.getScoreFeasible(famData, subData, ocA)))

    # write FINAL sample submission file
    h.writeSampleSub(subData, 'sampleExample_0')
