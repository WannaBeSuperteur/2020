import deepLearning_main as DL
import deepLearning_GPU_helper as helper
import randomData as RD
import algorithms as ALGO
import saveFiles as SF
import numpy as np
import random

# 배열 복사
def arrayCopy1d(array):
    copiedArray = []
    for i in range(len(array)): copiedArray.append(array[i])
    return copiedArray

# 배열의 각 열을 주어진 평균과 표준편차로 표준화
def normalize(array, means, stddevs, cols):
    copiedArray = helper.arrayCopy(array)
    rows = len(copiedArray)
    
    for i in range(cols):
        for j in range(rows):
            try:
                if stddevs[i] == 0: stddevs[i] = 1.0 # stddev가 0이면 조절
                copiedArray[j][i] = (copiedArray[j][i] - means[i])/stddevs[i]
            except:
                copiedArray[j][i] = (copiedArray[j][i] - means)/stddevs

    return copiedArray

# 배열에서 최솟값 반환
def indexMin(array):
    minIndex = 0
    minValue = array[0]

    for i in range(len(array)):
        if array[i] < minValue:
            minValue = array[i]
            minIndex = i

    return minIndex
    
if __name__ == '__main__':

    ## 파일 읽기
    f = open('input_output_info.txt', 'r')
    ioInfo = f.readlines()
    f.close()

    trainInputFileName = ioInfo[0].split('\n')[0] # train input data file (line 1)
    trainOutputFileName_0 = ioInfo[1].split('\n')[0] # train output data file - comparison (line 2)
    trainOutputFileName_1 = ioInfo[2].split('\n')[0] # train output data file - move (line 3)
    trainOutputFileName_2 = ioInfo[3].split('\n')[0] # train output data file - total cost (line 4)
    trainNormalizedOutputFileName_0 = trainOutputFileName_0.split('.')[0] + '_normalized.txt'
    trainNormalizedOutputFileName_1 = trainOutputFileName_1.split('.')[0] + '_normalized.txt'
    trainNormalizedOutputFileName_2 = trainOutputFileName_2.split('.')[0] + '_normalized.txt'
    
    testInputFileName = ioInfo[4].split('\n')[0] # test input data file (line 5)
    testOutputFileName_0 = ioInfo[5].split('\n')[0] # test output data file - comparison (line 6)
    testOutputFileName_1 = ioInfo[6].split('\n')[0] # test output data file - move (line 7)
    testOutputFileName_2 = ioInfo[7].split('\n')[0] # test output data file - total cost (line 8)
    testNormalizedOutputFileName_0 = testOutputFileName_0.split('.')[0] + '_normalized.txt'
    testNormalizedOutputFileName_1 = testOutputFileName_1.split('.')[0] + '_normalized.txt'
    testNormalizedOutputFileName_2 = testOutputFileName_2.split('.')[0] + '_normalized.txt'

    size = int(ioInfo[8].split('\n')[0]) # size of image (line 9)

    trainImgs = [] # train image input
    trainOutputs_0 = [] # train output - comparison
    trainOutputs_1 = [] # train output - move
    trainOutputs_2 = [] # train output - total cost
    
    testImgs = [] # test image input
    testOutputs_0 = [] # test output - comparison
    testOutputs_1 = [] # test output - move
    testOutputs_2 = [] # test output - total cost

    # 학습 데이터 개수(gt=0, gt=1, gt=2)
    trainGT0 = 300
    trainGT1 = 450
    trainGT2 = 750

    # 테스트 데이터 개수(gt=0, gt=1, gt=2)
    testGT0 = 100
    testGT1 = 150
    testGT2 = 250

    ## 학습 데이터 생성
    for i in range(trainGT0 + trainGT1 + trainGT2):
        if i % 10 == 0: print(i)

        # set graph type
        if i < trainGT0: gt = 0 # trainGT0 개의 상수함수 그래프 생성
        elif i < trainGT0 + trainGT1: gt = 1 # trainGT1 개의 일차함수 그래프 생성
        else: gt = 2 # trainGT2 개의 이차함수 그래프 생성

        # display setting
        display = False
        # if i == 0 or i == trainGT0 or i == trainGT0 + trainGT1: display = True

        # test input values
        gdd = random.random() * 50
        udd = random.random() * 50
        
        A = random.random()*2 - 1
        B = A * random.random() * (-2)
        C = random.random()
        
        (ys, img) = RD.createValues(gaussianDistDev=gdd, uniformDistDev=udd, a=A, b=B, c=C,
                     graphType=gt, display=False, imgHeight=10, imgWidth=10)
        
        trainImgs.append(img)

        # train output values
        # source: https://github.com/ztgu/sorting_algorithms_py
        (compsB, movesB) = ALGO.bubble(arrayCopy1d(ys))
        (compsH, movesH) = ALGO.heap(arrayCopy1d(ys))
        (compsI, movesI) = ALGO.insertion(arrayCopy1d(ys))
        (compsM, movesM) = ALGO.merge(arrayCopy1d(ys))
        (compsQ, movesQ) = ALGO.quick(arrayCopy1d(ys))
        (compsS, movesS) = ALGO.selection(arrayCopy1d(ys))

        # train output - comparison
        trainOutputs_0.append([compsB, compsH, compsI, compsM, compsQ, compsS])

        # train output - move
        trainOutputs_1.append([movesB, movesH, movesI, movesM, movesQ, movesS])

        # train output - total cost
        trainOutputs_2.append([compsB + movesB, compsH + movesH, compsI + movesI,
                               compsM + movesM, compsQ + movesQ, compsS + movesS])

    # 학습 데이터의 평균 및 표준편차 구하기
    # 테스트 시에도 학습 데이터 기준으로 평균 및 표준편차 적용
    trainO_0_mean = np.mean(trainOutputs_0, axis=0) # trainOutputs_0의 각 열의 평균
    trainO_1_mean = np.mean(trainOutputs_1, axis=0) # trainOutputs_1의 각 열의 평균
    trainO_2_mean = np.mean(trainOutputs_2, axis=0) # trainOutputs_2의 각 열의 평균
    trainO_0_std = np.std(trainOutputs_0, axis=0) # trainOutputs_0의 각 열의 표준편차
    trainO_1_std = np.std(trainOutputs_1, axis=0) # trainOutputs_1의 각 열의 표준편차
    trainO_2_std = np.std(trainOutputs_2, axis=0) # trainOutputs_2의 각 열의 표준편차

    # write to file
    SF.saveAsFile(trainImgs, 2, trainInputFileName) # train input
    SF.saveAsFile(trainOutputs_0, 1, trainOutputFileName_0) # train output (comparison)
    SF.saveAsFile(trainOutputs_1, 1, trainOutputFileName_1) # train output (move)
    SF.saveAsFile(trainOutputs_2, 1, trainOutputFileName_2) # train output (total cost)
    SF.saveAsFile(normalize(trainOutputs_0, trainO_0_mean, trainO_0_std, 6), 1, trainNormalizedOutputFileName_0) # normalized train output (comparison)
    SF.saveAsFile(normalize(trainOutputs_1, trainO_1_mean, trainO_1_std, 6), 1, trainNormalizedOutputFileName_1) # normalized train output (move)
    SF.saveAsFile(normalize(trainOutputs_2, trainO_2_mean, trainO_2_std, 6), 1, trainNormalizedOutputFileName_2) # normalized train output (total cost)

    ## 테스트 데이터 생성
    for i in range(testGT0 + testGT1 + testGT2):
        if i % 10 == 0: print(i)

        # set graph type
        if i < testGT0: gt = 0 # testGT0 개의 상수함수 그래프 생성
        elif i < testGT0 + testGT1: gt = 1 # testGT1 개의 일차함수 그래프 생성
        else: gt = 2 # testGT2 개의 이차함수 그래프 생성

        # display setting
        display = False
        # if i == 0 or i == testGT0 or i == testGT0 + testGT1: display = True

        # test input values
        gdd = random.random() * 50
        udd = random.random() * 50
        
        A = random.random()*2 - 1
        B = A * random.random() * (-2)
        C = random.random()
        
        (ys, img) = RD.createValues(gaussianDistDev=gdd, uniformDistDev=udd, a=A, b=B, c=C,
                     graphType=gt, display=False, imgHeight=10, imgWidth=10)
        
        testImgs.append(img)

        # train output values
        # source: https://github.com/ztgu/sorting_algorithms_py
        (compsB, movesB) = ALGO.bubble(arrayCopy1d(ys))
        (compsH, movesH) = ALGO.heap(arrayCopy1d(ys))
        (compsI, movesI) = ALGO.insertion(arrayCopy1d(ys))
        (compsM, movesM) = ALGO.merge(arrayCopy1d(ys))
        (compsQ, movesQ) = ALGO.quick(arrayCopy1d(ys))
        (compsS, movesS) = ALGO.selection(arrayCopy1d(ys))
        
        # test output - comparison
        testOutputs_0.append([compsB, compsH, compsI, compsM, compsQ, compsS])

        # test output - move
        testOutputs_1.append([movesB, movesH, movesI, movesM, movesQ, movesS])

        # test output - total cost
        testOutputs_2.append([compsB + movesB, compsH + movesH, compsI + movesI,
                              compsM + movesM, compsQ + movesQ, compsS + movesS])

    # write to file
    SF.saveAsFile(testImgs, 2, testInputFileName) # test input
    SF.saveAsFile(testOutputs_0, 1, testOutputFileName_0) # test output (comparison)
    SF.saveAsFile(testOutputs_1, 1, testOutputFileName_1) # test output (move)
    SF.saveAsFile(testOutputs_2, 1, testOutputFileName_2) # test output (total cost)
    SF.saveAsFile(normalize(testOutputs_0, trainO_0_mean, trainO_0_std, 6), 1, testNormalizedOutputFileName_0) # normalized test output (comparison)
    SF.saveAsFile(normalize(testOutputs_1, trainO_1_mean, trainO_1_std, 6), 1, testNormalizedOutputFileName_1) # normalized test output (move)
    SF.saveAsFile(normalize(testOutputs_2, trainO_2_mean, trainO_2_std, 6), 1, testNormalizedOutputFileName_2) # normalized test output (total cost)

    ## 딥러닝 실시
    ## 관련 학습/테스트 입출력 데이터 파일이 이미 존재하면 딥러닝에 의한 예측값만 파일로 쓰기 
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    # comparison
    print('\n ### deep learning for COMPARISON ###')
    DL.deepLearning(trainInputFileName, trainNormalizedOutputFileName_0,
                    testInputFileName, testNormalizedOutputFileName_0,
                    size, deviceName, epoch, printed, 'test0_comparison')

    # move
    print('\n\n\n ### deep learning for MOVE ###')
    DL.deepLearning(trainInputFileName, trainNormalizedOutputFileName_1,
                    testInputFileName, testNormalizedOutputFileName_1,
                    size, deviceName, epoch, printed, 'test1_move')

    # total cost
    print('\n\n\n ### deep learning for TOTAL COST ###')
    DL.deepLearning(trainInputFileName, trainNormalizedOutputFileName_2,
                    testInputFileName, testNormalizedOutputFileName_2,
                    size, deviceName, epoch, printed, 'test2_totalCost')

    ## 딥러닝 결과로부터 loss 계산 : mean absolute and mean square error
    # total count (testGT의 전체 수 * 알고리즘 수)
    totalTestMaps = testGT0 + testGT1 + testGT2 # testGT의 전체 수
    algorithms = 6
    totalCount = totalTestMaps * algorithms
    
    # read files
    f0 = helper.getDataFromFile(testNormalizedOutputFileName_0, None)
    f0P = helper.getDataFromFile(testNormalizedOutputFileName_0.split('.')[0] + '_prediction.txt', None)

    f1 = helper.getDataFromFile(testNormalizedOutputFileName_1, None)
    f1P = helper.getDataFromFile(testNormalizedOutputFileName_1.split('.')[0] + '_prediction.txt', None)

    f2 = helper.getDataFromFile(testNormalizedOutputFileName_2, None)
    f2P = helper.getDataFromFile(testNormalizedOutputFileName_2.split('.')[0] + '_prediction.txt', None)
    
    # mean absolute error (MAE) : (1/n) * Sum(i=1,n)|fi-oi|
    # mean square error (MSE) : (1/n) * Sum(i=1,n)(fi-oi)^2
    MAE0 = 0.0 # MAE for comparison
    MAE1 = 0.0 # MAE for move
    MAE2 = 0.0 # MAE for total cost
    MSE0 = 0.0 # MSE for comparison
    MSE1 = 0.0 # MSE for move
    MSE2 = 0.0 # MSE for total cost
    accuracy0 = 0.0 # accuracy for comparison
    accuracy1 = 0.0 # accuracy for move
    accuracy2 = 0.0 # accuracy for total cost
    
    minIndices = '0=bubble, 1=heap, 2=insertion, 3=merge, 4=quick, 5=selection\nbestAlgorithmPrediction,bestAlgorithmReal\ncomp\tmove\ttotal\n' # min indices info

    for i in range(totalTestMaps):
        for j in range(algorithms):
            MAE0 += abs(f0[i][j] - f0P[i][j]) # mean absolute error
            MAE1 += abs(f1[i][j] - f1P[i][j])
            MAE2 += abs(f2[i][j] - f2P[i][j])
            MSE0 += pow(f0[i][j] - f0P[i][j], 2) # mean square error
            MSE1 += pow(f1[i][j] - f1P[i][j], 2)
            MSE2 += pow(f2[i][j] - f2P[i][j], 2)

        # accuracy
        if indexMin(f0[i]) == indexMin(f0P[i]): accuracy0 += 1
        if indexMin(f1[i]) == indexMin(f1P[i]): accuracy1 += 1
        if indexMin(f2[i]) == indexMin(f2P[i]): accuracy2 += 1

        # add result for calculating accuracy
        minIndices += str(indexMin(f0P[i])) + ',' + str(indexMin(f0[i])) + '\t'
        minIndices += str(indexMin(f1P[i])) + ',' + str(indexMin(f1[i])) + '\t'
        minIndices += str(indexMin(f2P[i])) + ',' + str(indexMin(f2[i])) + '\n'

    # total -> mean
    MAE0 /= totalCount
    MAE1 /= totalCount
    MAE2 /= totalCount
    MSE0 /= totalCount
    MSE1 /= totalCount
    MSE2 /= totalCount
    accuracy0 /= totalTestMaps
    accuracy1 /= totalTestMaps
    accuracy2 /= totalTestMaps

    # save result of error
    errorResult =  'MAE       for comparison : ' + str(round(MAE0, 8)) + '\n'
    errorResult += 'MAE       for move       : ' + str(round(MAE1, 8)) + '\n'
    errorResult += 'MAE       for total cost : ' + str(round(MAE2, 8)) + '\n'
    errorResult += 'MSE       for comparison : ' + str(round(MSE0, 8)) + '\n'
    errorResult += 'MSE       for move       : ' + str(round(MSE1, 8)) + '\n'
    errorResult += 'MSE       for total cost : ' + str(round(MSE2, 8)) + '\n'
    errorResult += 'accuracy0 for comparison : ' + str(round(accuracy0, 4)) + '\n'
    errorResult += 'accuracy1 for move       : ' + str(round(accuracy1, 4)) + '\n'
    errorResult += 'accuracy2 for total cost : ' + str(round(accuracy2, 4)) + '\n'

    fResult = open('errorResult.txt', 'w')
    fResult.write(errorResult)
    fResult.close()

    fMinIndices = open('minIndices.txt', 'w')
    fMinIndices.write(minIndices)
    fMinIndices.close()

# 오류해결
# DL.deepLearning에서
#  File "C:\Users\hskim\2020\2020_02_32618\deepLearning_GPU_helper.py", line 106, in getDataFromFile
#    thisLineSplit = fLines[i * height + j].split('\n')[0].split('\t')
# IndexError: list index out of range
# ALGO의 값이 계속 누적되는 오류해결
