import os

version = input('version')
size = int(input('size'))
nTrain = int(input('number of train data'))
nTest = int(input('number of test data'))

for i in range(2):

    # open file
    f = open('optiInfoForMap_' + str(i) + '_ver' + str(version) + '.txt')
    fr = f.readlines()
    f.close()

    train_sumMaxThroughput = 0.0
    test_sumMaxThroughput = 0.0

    # read lines
    # for training data
    for j in range(nTrain):
        train_sumMaxThroughput += float(fr[j * (size+1)].split(' ')[5])

    # for test data
    for j in range(nTest):
        test_sumMaxThroughput += float(fr[(nTrain + j) * (size+1)].split(' ')[5])

    # print data
    all_sumMaxThroughput = train_sumMaxThroughput + test_sumMaxThroughput
    
    print('\n <<< problem ' + str(i) + ' >>>')
    print('sum train: ' + str(round(train_sumMaxThroughput, 6)))
    print('avg train: ' + str(round(train_sumMaxThroughput / nTrain, 6)))
    print('\nsum test : ' + str(round(test_sumMaxThroughput, 6)))
    print('avg test : ' + str(round(test_sumMaxThroughput / nTest, 6)))
    print('\nsum all  : ' + str(round(all_sumMaxThroughput, 6)))
    print('avg all  : ' + str(round(all_sumMaxThroughput / (nTrain + nTest), 6)))
