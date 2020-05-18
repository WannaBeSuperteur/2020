# 목적:
# WPCN에서 HAP(=drone)의 위치와 충전 시간 할당 방법을 최적화한다.

# map.txt의 구성:
# (맵 세로길이) (맵 가로길이) (wireless device의 개수) (train 개수) (test 개수)

# 입력: 전체 map (size가 6x6 이상인 경우 CNN 적용)
# 출력:
# 1) SUM-THROUGHPUT MAXIMIZATION의 경우
#     sum-throughput이 가장 커지는 좌표 2개 (y, x좌표 순)
#     sum-throughput이 가장 커지는 할당 시간 (HAP의 할당 시간 1-x)
#     -> 각 wireless device의 충전 시간은 x(1-x), x^2(1-x), x^3(1-x), ...
# 2) COMMON THROUGHPUT MAXIMIZATION의 경우
#     common-throughput (=minimum throughput)이 가장 커지는 좌표 2개 (y, x좌표 순)
#     common-throughput (=minimum throughput)이 가장 커지는 할당 시간 (HAP의 할당 시간 1-x)
#     -> 각 wireless device의 충전 시간은 x(1-x), x^2(1-x), x^3(1-x), ...
#
# convex/concave function이므로 위치가 바뀜에 따라 THROUGHPUT의 값이 convex/concave하게 바뀜
# -> 따라서 딥러닝을 적용하기에 비교적 용이하다고 판단 

import math
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

import sys
sys.path.insert(0, '../DQN')
import Qlearning_deep
import Qlearning_deep_helper

import WPCN_helper

# WPCN에서 throughput의 값 구하기
# screen: 스크린(맵)
# y     : HAP의 y좌표
# x     : HAP의 x좌표
# wdList: wireless device의 목록
# reverse: True 또는 False
#          --------
#          True이면 wdList를 distance의 제곱의 역순으로 정렬하여 먼 WD에 시간을 많이 할당
#          Common throughput maximization (throughput의 최솟값을 최대화) 에서 사용
#          --------
#          False이면 wdList를 distance의 제곱 순으로 정렬하여 가까운 WD에 시간을 많이 할당
#          Sum-throughput maximization (throughput의 합) 에서 사용

# 진행
if __name__ == '__main__':

    # 사용자 입력
    problemNo = int(input('0->sum throughtput maximization, 1->common throughtput maximization'))
    readOrWrite = int(input('0->read files, 1->create new files'))

    # 0. 파일을 읽어서 size 값과 train 개수, test 개수 구하기
    # 파일 형식 : (맵 세로길이) (맵 가로길이) (wireless device의 개수) (train 개수) (test 개수)
    # train 개수: 트레이닝할 데이터(맵)의 개수
    # test 개수 : 테스트할 데이터(맵)의 개수
    file = open('map.txt', 'r')
    read = file.readlines()
    readSplit = read[0].split(' ')
    
    height = int(readSplit[0]) # 맵의 세로길이 (= 가로길이 = size)
    size = height
    numTrain = int(readSplit[3]) # 트레이닝할 데이터(맵)의 개수
    numTest = int(readSplit[4]) # 테스트할 데이터(맵)의 개수
    
    file.close()
    
    inputs = [] # 딥러닝 입력값 저장
    outputs = [] # 딥러닝 출력값 저장
    
    isFile = False # valueMaze_X.json 학습 모델 파일이 있는가?

    # 입력 정보
    originalScreen = [] # 각 맵의 스크린
    wdList = [] # 각 맵에서 wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]

    # 출력 정보
    optiY = [] # 각 맵에서 throughput (sum 또는 common)이 최대가 되는 HAP의 y좌표
    optiX = [] # 각 맵에서 throughput (sum 또는 common)이 최대가 되는 HAP의 x좌표
    optiHT = [] # 각 맵에서 throughput (sum 또는 common)이 최대가 되는 HAP의 충전 할당 시간

    # 평가 결과 저장
    RL = []

    # 1. 트레이닝 및 테스트용 맵 생성
    # 맵의 구성: .(공간), H(HAP), W(wireless device)
    if readOrWrite == 1: # 맵 파일 쓰고 그 맵으로 트레이닝, 테스트
        print('generating maps...')
        for i in range(numTrain + numTest):
            initScreen_ = WPCN_helper.initScreen(size, False)
            
            originalScreen.append(initScreen_[0])
            wdList.append(initScreen_[2])

            # 맵 파일 작성
            f = open('DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'w')
            for j in range(size):
                for k in range(size):
                    if originalScreen[i][j][k] == -1: f.write('W') # wireless device
                    elif originalScreen[i][j][k] == 0: f.write('.') # space
                    elif originalScreen[i][j][k] == 1: f.write('H') # HAP (not exist in the original map)
                f.write('\n')
            f.close()
            
    else: # 기존 맵 파일 읽어서 기존 맵으로 트레이닝, 테스트
        print('reading maps...')
        for i in range(numTrain + numTest):
            f = open('DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'r')
            map_ = f.readlines() # 맵을 나타낸 배열
            file.close()
            for j in range(len(map_)): map_[j] = map_[j].replace('\n', '') # 각 줄마다 개행문자 제거

            # originalScreen과 wdList 읽어오기
            wdListTemp = []
            thisScreen = [[0] * size for j in range(size)] # originalScreen에 추가할 맵(스크린)
            for j in range(size):
                for k in range(size):
                    if map_[j][k] == 'W':
                        thisScreen[j][k] = -1 # wireless device
                        wdListTemp.append([j, k]) # wdListTemp에 추가
                    elif map_[j][k] == '.': thisScreen[j][k] = 0 # space
                    elif map_[j][k] == 'H': thisScreen[j][k] = 1 # HAP (not exist in the original map)
                    
            originalScreen.append(thisScreen)
            wdList.append(wdListTemp)

    # print(wdList)

    # 2. 트레이닝용 맵에서 최적의 HAP의 위치(y좌표, x좌표 순) 및 할당 시간(1-x) 찾기
    # y좌표, x좌표는 1 간격, 할당 시간은 0.01 간격
    for i in range(numTrain):
        if i % 10 == 0: print('finding max throughput for map ' + str(i) + '...')
        
        optiY_ = 0 # throughput (sum 또는 common)이 최대가 되는 HAP의 y좌표
        optiX_ = 0 # throughput (sum 또는 common)이 최대가 되는 HAP의 x좌표
        optiHT_ = 0 # throughput (sum 또는 common)이 최대가 되는 HAP의 충전 할당 시간
            
        maxThroughput_ = 0.0 # throughput (sum 또는 common)의 최댓값

        # 최적의 HAP의 위치 및 할당 시간 찾기
        for y in range(size):
            for x in range(size):
                HAPtimeList = []
                for j in range(99): HAPtimeList.append(0.01 + 0.01 * j) # [0.01, 0.02, ..., 0.99]
                
                # HAPtime = 1-x where each wireless device charging time is x(1-x), ...
                for HAPtime in HAPtimeList:
                    throughput = WPCN_helper.getThroughput(wdList[i], [y, x], size, HAPtime, problemNo)
                    # print(throughput)

                    # throughput 최고기록을 갱신한 경우 업데이트
                    if throughput > maxThroughput_:
                        optiY_ = y
                        optiX_ = x
                        optiHT_ = HAPtime
                        maxThroughput_ = throughput

        # print('\nmax throughput: \n' + str(maxThroughput_))

        # 출력값 배열에 추가
        optiY.append(Qlearning_deep_helper.sigmoid(optiY_))
        optiX.append(Qlearning_deep_helper.sigmoid(optiX_))
        optiHT.append(Qlearning_deep_helper.sigmoid(optiHT_))

    # optiY, optiX, optiHT 출력
    print('')
    for i in range(numTrain):
        optiY_IS = round(Qlearning_deep_helper.invSigmoid(optiY[i]), 1)
        optiX_IS = round(Qlearning_deep_helper.invSigmoid(optiX[i]), 1)
        optiHT_IS = round(Qlearning_deep_helper.invSigmoid(optiHT[i]), 3)
        
        print('training data (sigmoid not applied) for map ' + str(i) + ': '
              + str(optiY_IS) + ', ' + str(optiX_IS) + ', ' + str(optiHT_IS)) 
    print('')

    # 3. 트레이닝용 맵의 정보를 입력값과 출력값 배열에 넣기
    print('make input and output data...')
    
    for i in range(numTrain):
        originalScreen_ = Qlearning_deep_helper.arrayCopy(originalScreen[i]) # copy array from originalScreen
        originalScreen_ = Qlearning_deep_helper.arrayCopyFlatten(originalScreen_, 0, size, 0, size, None) # flatten
        # print(originalScreen_)
        inputs.append(originalScreen_)
        outputs.append([optiY[i], optiX[i], optiHT[i]])

    # 4~7. 학습 및 테스트는 주어진 입출력 데이터를 이용하여 계속 반복
    while True:
        epoch = int(input('epoch'))
        dropout = float(input('dropout'))

        # 4. 모델 정의
        if size >= 10: # 학습 영역의 크기가 10 이상
            NN = [tf.keras.layers.Reshape((size, size, 1), input_shape=(size*size,)),
                    keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(size, size, 1), activation='relu'),
                    keras.layers.MaxPooling2D(pool_size=2),
                    keras.layers.Dropout(dropout),
                    keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(dropout),
                    keras.layers.Dense(40, activation='relu'),
                    keras.layers.Dense(3, activation='sigmoid')]
            
        elif size >= 6: # 학습 영역의 크기가 6 이상 10 미만
            NN = [tf.keras.layers.Reshape((size, size, 1), input_shape=(size*size,)),
                    keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(size, size, 1), activation='relu'),
                    keras.layers.MaxPooling2D(pool_size=2),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(dropout),
                    keras.layers.Dense(40, activation='relu'),
                    keras.layers.Dense(3, activation='sigmoid')]
            
        else: # 학습 영역의 크기가 6 미만
            NN = [tf.keras.layers.Flatten(input_shape=(size*size,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(3, activation='sigmoid')]

        # 옵티마이저
        op = tf.keras.optimizers.Adam(0.001)

        # 5. 딥러닝 학습시키기
        lc = 'mean_squared_error' # loss calculator
        print('training...')
        Qlearning_deep.deepQlearning(0, NN, op, lc, inputs, outputs, None, None, 'WPCNdeep', [epoch, None], None, None, False, False, True)

        # 6. 테스트 결과 확인하고 정답과 비교하기
        newModel = Qlearning_deep.deepLearningModel('WPCNdeep_0', False)
        sumTestThroughput = 0.0 # test throughput의 합계
        sumCorrectMaxThroughput = 0.0 # 정답의 throughput의 합계 (training data를 생성할 때와 같은 방법으로 최대 throughput 확인)

        print('testing...')
        
        for i in range(numTest):
            testScreen = originalScreen[numTrain + i] # 테스트할 스크린
            testOutput = Qlearning_deep.modelOutput(newModel, [testScreen]) # 테스트 결과
            testOutputLayer = testOutput[len(testOutput)-1] # 테스트 결과의 output layer의 값

            # 6-0. 테스트 결과 확인
            # 출력값 받아오기
            optiY_test = Qlearning_deep_helper.invSigmoid(testOutputLayer[0][0]) # 테스트 결과로 나온 throughput 최대화 y좌표
            optiX_test = Qlearning_deep_helper.invSigmoid(testOutputLayer[0][1]) # 테스트 결과로 나온 throughput 최대화 x좌표
            optiHT_test = Qlearning_deep_helper.invSigmoid(testOutputLayer[0][2]) # 테스트 결과로 나온 throughput 최대화 충전시간

            # 0 <= HAP time <= 1
            if optiHT_test < 0.0: optiHT_test = 0.0
            elif optiHT_test > 1.0: optiHT_test = 1.0

            # 출력값 중 HAP의 좌표를 정수로 조정
            optiY_test = int(round(optiY_test, 0))
            if optiY_test >= size: optiY_test = size-1
            elif optiY_test < 0: optiY_test = 0
            
            optiX_test = int(round(optiX_test, 0))
            if optiX_test >= size: optiX_test = size-1
            elif optiX_test < 0: optiX_test = 0

            # 테스트 결과 받아오기
            throughput = WPCN_helper.getThroughput(wdList[numTrain + i], [optiY_test, optiX_test], size, optiHT_test, problemNo)
            sumTestThroughput += throughput

            # 6-1. 정답과 비교
            # 최적의 HAP의 위치 및 할당 시간 찾기
            correctMaxThroughput = 0.0 # HAP의 위치 및 할당 시간에 따른 throughput의 최댓값
            for y in range(size):
                for x in range(size):
                    HAPtimeList = []
                    for j in range(99): HAPtimeList.append(0.01 + 0.01 * j) # [0.01, 0.02, ..., 0.99]
                    
                    # HAPtime = 1-x where each wireless device charging time is x(1-x), x^2(1-x), ...
                    for HAPtime in HAPtimeList:
                        thisThroughput = WPCN_helper.getThroughput(wdList[numTrain + i], [y, x], size, HAPtime, problemNo)

                        # throughput 최고기록을 갱신한 경우 업데이트
                        if thisThroughput > correctMaxThroughput: correctMaxThroughput = thisThroughput

            sumCorrectMaxThroughput += correctMaxThroughput

            # 정답과 비교
            print('test throughput for map ' + str(i) + ' [Y=' + str(optiY_test) + ' X=' + str(optiX_test) + ' HT=' + str(optiHT_test) + '] : '
                  + str(round(throughput, 6)) + ' / ' + str(round(correctMaxThroughput, 6)) + ' ('
                  + str(round(100.0 * throughput / correctMaxThroughput, 6)) + ' %)')

        # 7. 평가
        percentage = round(100.0 * sumTestThroughput / sumCorrectMaxThroughput, 6)
        print('size:' + str(size) + ' train:' + str(numTrain) + ' test:' + str(numTest)
              + ' problem(0=Sum,1=Common):' + str(problemNo) + ' epoch:' + str(epoch) + ' dropout:' + str(round(dropout, 3)))
        print('total test throughput: ' + str(round(sumTestThroughput, 6)))
        print('total max  throughput: ' + str(round(sumCorrectMaxThroughput, 6)))
        print('percentage           : ' + str(percentage) + ' %')

        RL.append([size, numTrain, numTest, problemNo, epoch, dropout, percentage])

        # 현재까지의 평가 결과 출력
        print('\n <<< ESTIMATION RESULT >>>')
        print('No.\tsize\ttrain\ttest\tproblem\tepoch\tdropout\tpercentage')
        
        for i in range(len(RL)):
            print(str(i) + '\t' + str(RL[i][0]) + '\t' + str(RL[i][1]) + '\t' + str(RL[i][2]) + '\t'
                  + str(RL[i][3]) + '\t' + str(RL[i][4]) + '\t'
                  + str(round(RL[i][5], 3)) + '\t' + (' ' if round(RL[i][6], 4) < 10.0 else '') + str(round(RL[i][6], 4)))
        print('\n')
