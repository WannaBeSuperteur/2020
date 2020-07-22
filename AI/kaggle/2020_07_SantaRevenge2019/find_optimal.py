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

    # calculate cost
    for i in range(1000):

        subDataCopy = []
        leastScore = 2147483647
        
        for j in range(50):
            subDataCopy.append(h.arrayCopy(subData))
            for k in range(FAMILIES):
                if random.randint(0, 30) == 5:
                    newChoice = random.randint(0, 9)
                    subDataCopy[j][k][1] = famData[k][1+newChoice]
            
            score = h.getScore(famData, subDataCopy[j])
            
            if score < leastScore:
                leastScore = score
                leastIndex = j

        if leastScore < lastScore:
            subData = subDataCopy[leastIndex]
            print('new subData(' + str(i) + '): at index ' + str(leastIndex) + ' -> score: ' + str(leastScore))
            # h.writeSampleSub(subData, 'famData_' + str(i))
            lastScore = leastScore

    # write FINAL sample submission file
    h.writeSampleSub(subData, 'sampleExample')
