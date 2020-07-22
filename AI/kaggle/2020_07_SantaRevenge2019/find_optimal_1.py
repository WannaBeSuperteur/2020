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
    options = []
    for i in range(len(subData)): options.append(0)
        
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

        offset = random.randint(0, FAMILIES-1)
        for i in range(offset, len(subData) + offset):
            val = i % FAMILIES
            
            if subData[val][1] == maxIndex:
                option = -1
                if subData[val][1] == famData[val][1] and famData[val][2] != maxIndex:
                    option = 0
                    subData[val][1] = famData[val][2]
                elif subData[val][1] == famData[val][2] and famData[val][1] != maxIndex:
                    option = 0
                    subData[val][1] = famData[val][1]
                elif subData[val][1] == famData[val][2] and famData[val][3] != maxIndex:
                    option = 1
                    subData[val][1] = famData[val][3]
                elif subData[val][1] == famData[val][3] and famData[val][2] != maxIndex:
                    option = 0
                    subData[val][1] = famData[val][2]
                elif subData[val][1] == famData[val][3] and famData[val][4] != maxIndex:
                    option = 2
                    subData[val][1] = famData[val][4]
                elif subData[val][1] == famData[val][4] and famData[val][3] != maxIndex:
                    option = 1
                    subData[val][1] = famData[val][3]
                elif subData[val][1] == famData[val][4] and famData[val][5] != maxIndex:
                    option = 3
                    subData[val][1] = famData[val][5]
                elif subData[val][1] == famData[val][5] and famData[val][4] != maxIndex:
                    option = 2
                    subData[val][1] = famData[val][4]
                elif subData[val][1] == famData[val][5] and famData[val][6] != maxIndex:
                    option = 4
                    subData[val][1] = famData[val][6]
                elif subData[val][1] == famData[val][6] and famData[val][5] != maxIndex:
                    option = 3
                    subData[val][1] = famData[val][5]
                elif subData[val][1] == famData[val][6] and famData[val][7] != maxIndex:
                    option = 5
                    subData[val][1] = famData[val][7]
                elif subData[val][1] == famData[val][7] and famData[val][6] != maxIndex:
                    option = 4
                    subData[val][1] = famData[val][6]
                elif subData[val][1] == famData[val][7] and famData[val][8] != maxIndex:
                    option = 6
                    subData[val][1] = famData[val][8]
                elif subData[val][1] == famData[val][8] and famData[val][7] != maxIndex:
                    option = 5
                    subData[val][1] = famData[val][7]
                elif subData[val][1] == famData[val][8] and famData[val][9] != maxIndex:
                    option = 7
                    subData[val][1] = famData[val][9]
                elif subData[val][1] == famData[val][9] and famData[val][8] != maxIndex:
                    option = 6
                    subData[val][1] = famData[val][8]
                elif subData[val][1] == famData[val][9] and famData[val][10] != maxIndex:
                    option = 8
                    subData[val][1] = famData[val][10]
                elif subData[val][1] == famData[val][10] and famData[val][9] != maxIndex:
                    option = 7
                    subData[val][1] = famData[val][9]
                elif subData[val][1] == famData[val][10]:
                    option = 9
                    subData[val][1] = minIndex
                #print(option)

                if option > 0: options[i % FAMILIES] = option
                break

        prefCost = h.prefCost(famData, subData, ocA)
        lastScore = h.getScoreFeasible(famData, subData, ocA)
        if prefCost > 1000000: break

        print('val: ' + str(maxVal) + ' ' + str(minVal) + ' / index: ' + str(maxIndex) + ' ' + str(minIndex) +
              ' / cost: ' + str(prefCost) + ' score: ' + str(lastScore) + ' / sum: ' +
              str(sum(options[0:1000])) + ' ' + str(sum(options[1000:2000])) + ' ' + str(sum(options[2000:3000])) +
              str(sum(options[3000:4000])) + ' ' + str(sum(options[4000:5000])) + ' ' + str(sum(options[5000:6000])))

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
