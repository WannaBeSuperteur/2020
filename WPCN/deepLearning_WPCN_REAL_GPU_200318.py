# 목적:
# WPCN에서 HAP(=drone)의 위치와 충전 시간 할당 방법을 최적화한다.

# map.txt의 구성:
# (맵 세로길이) (맵 가로길이) (wireless device의 개수) (train 개수) (test 개수)

# 입력: 전체 map (size가 6x6 이상인 경우 CNN 적용)
# 출력: 전체 map에서 HAP의 각 좌표에서의 throughput을 나타낸 맵
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
import Qlearning_deep_GPU
import Qlearning_deep_GPU_helper

import WPCN_helper_REAL

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
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    
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
    outputScreen = [] # 각 맵에서의 각 좌표에서의 throughput을 나타낸 맵, [[[thrput(0,0), thrput(0,1), ...], [thrput(1,0), ...], ...], (so on)]

    # 평가 결과 저장
    RL = []

    # 1. 트레이닝 및 테스트용 맵 생성
    # 맵의 구성: .(공간), H(HAP), W(wireless device)
    if readOrWrite == 1: # 맵 파일 쓰고 그 맵으로 트레이닝, 테스트
        print('generating maps...')
        for i in range(numTrain + numTest):
            initScreen_ = WPCN_helper_REAL.initScreen(size, False)
            
            originalScreen.append(initScreen_[0])
            wdList.append(initScreen_[2])

            # 맵 파일 작성
            f = open('originalMaps/DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'w')
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
            f = open('originalMaps/DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'r')
            map_ = f.readlines() # 맵을 나타낸 배열
            f.close()
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

    # 2. 트레이닝 및 테스트용 맵에서 각 x, y (HAP의 좌표)에 대해 최적의 할당 시간(HAPtime)을 찾아서,
    # HAP의 각 좌표에서의 throughput에 대한 맵 만들기 (optiInfoForMap_X.txt where X = 0 or 1)
    # y좌표, x좌표는 1 간격

    # 형식:
    # * (맵 번호) (HAPtime) (최적y) (최적x) (최대thrput)
    # (thrput at y=0,x=0) (thrput at y=0,x=1) ... (thrput at y=0,x=n)
    # (thrput at y=1,x=0) (thrput at y=1,x=1) ... (thrput at y=1,x=n)
    # ...                 ...                     ...
    # (thrput at y=n,x=0) (thrput at y=n,x=1) ... (thrput at y=n,x=n)
    toSave = ''
    
    lines = 0 # optiInfoForMap_X.txt 파일의 라인 개수 (X = 0 or 1)
    
    try: # optiInfoForMap 파일이 있으면 읽기
        f = open('optiInfoForMap_' + str(problemNo) + '.txt', 'r')
        optiInformation = f.readlines()
        f.close()

        lines = len(optiInformation)

        if len(optiInformation) < numTrain + numTest: # 완전한 정보가 없으면
            for i in range(lines): toSave += optiInformation[i] # toSave에 기존 저장된 정보 추가
            raiseError = 1 / 0 # 오류 발생시키기

        # 출력값 배열에 기존 파일에 있는 추가
        temp = [] # 출력값 배열에 추가할 데이터
        for i in range(lines):

            # '*' 기호 (새로운 맵)를 만나면 기존 temp 배열의 데이터를 저장하고 temp는 초기화
            if optiInformation[i][0] == '*':
                if len(temp) > 0: outputScreen.append(temp)
                temp = []
            # 그렇지 않으면 temp에 해당 행의 데이터를 추가
            elif len(optiInformation[i]) >= 3: # 공백 또는 개행만 있는 줄 제외
                temp.append(optiInformation[i].split(' '))
                for j in range(len(temp[len(temp)-1])): temp[len(temp)-1][j] = float(temp[len(temp)-1][j])
        
    except Exception as e: # optiInfoForMap 파일이 없으면 새로 생성하기
        print(e)
        print("can't read optiInfoForMap_" + str(problemNo) + ".txt")
        
        for i in range(lines, numTrain + numTest):
            print('finding max throughput for map ' + str(i) + '...')

            # HAP의 위치(각 좌표)에 따른 최적의 할당 시간을 찾아서 출력값에 추가
            temp = [[0] * size for _ in range(size)] # 출력값 배열에 추가할 데이터 (해당 x, y 좌표에서의 HAPtime에 따른 최대 throughput)

            optiY_ = 0 # throughput (sum 또는 common)이 최대가 되는 HAP의 y좌표
            optiX_ = 0 # throughput (sum 또는 common)이 최대가 되는 HAP의 x좌표
            maxThroughput_ = 0.0 # throughput (sum 또는 common)의 최댓값
            maxHAPtime_ = 0.0 # throughput (sum 또는 common)이 최대가 되기 위한 HAP에 할당되는 시간
            
            for y in range(size):
                for x in range(size):
                    (throughput, HAPtime) = WPCN_helper_REAL.getThroughput(wdList[i], [y, x], size, problemNo)
                    temp[y][x] = throughput

                    # throughput 최고기록을 갱신한 경우 업데이트
                    if throughput > maxThroughput_:
                        optiY_ = y
                        optiX_ = x
                        maxThroughput_ = throughput
                        maxHAPtime_ = HAPtime

            # toSave에 추가
            # * (맵 번호) (HAPtime) (최적y) (최적x) (최대thrput)
            # (thrput at y=0,x=0) (thrput at y=0,x=1) ... (thrput at y=0,x=n)
            # (thrput at y=1,x=0) (thrput at y=1,x=1) ... (thrput at y=1,x=n)
            # ...                 ...                     ...
            # (thrput at y=n,x=0) (thrput at y=n,x=1) ... (thrput at y=n,x=n)
            toSave += '* ' + str(i) + ' ' + str(maxHAPtime_) + ' ' + str(optiY_) + ' ' + str(optiX_) + ' ' + str(maxThroughput_) + '\n'
            for j in range(size):
                for k in range(size-1): toSave += str(temp[j][k]) + ' '
                toSave += str(temp[j][size-1]) + '\n'

            # 출력 및 toSave에 추가
            print('max throughput: ' + str(maxThroughput_) + ' axis: ' + str(optiY_) + ' ' + str(optiX_) + ' HAPtime: ' + str(maxHAPtime_))

            # 출력값 배열에 추가
            if i < numTrain: outputScreen.append(temp)

            # 파일로 저장
            optiInfo = open('optiInfoForMap_' + str(problemNo) + '.txt', 'w')
            optiInfo.write(toSave)
            optiInfo.close()

    # 3. 트레이닝용 맵의 정보를 입력값과 출력값 배열에 넣기
    print('make input and output data...')
    
    for i in range(numTrain):
        originalScreen_ = Qlearning_deep_GPU_helper.arrayCopy(originalScreen[i]) # copy array from originalScreen
        originalScreen_ = Qlearning_deep_GPU_helper.arrayCopyFlatten(originalScreen_, 0, size, 0, size, None) # flatten
        inputs.append(originalScreen_)

        # find max value in outputScreen[i] for common throughput maximization
        if problemNo == 1:
            maxVal = 0.0
            for j in range(size):
                for k in range(size):
                    if outputScreen[i][j][k] > maxVal: maxVal = outputScreen[i][j][k]
            for j in range(size):
                for k in range(size):
                    outputScreen[i][j][k] /= maxVal

        # 테스트 시 역sigmoid를 적용할 것이므로 먼저 outputScreen의 모든 원소에 sigmoid를 적용 (outputScreen의 값이 1 이상이 될수 있으므로)
        for j in range(size):
            for k in range(size): outputScreen[i][j][k] = Qlearning_deep_GPU_helper.sigmoid(outputScreen[i][j][k])
        
        outputs.append([outputScreen[i]])

    # 4~7. 학습 및 테스트는 주어진 입출력 데이터를 이용하여 계속 반복
    while True:
        epoch = input('epoch').split(' ') # 입력형식: epoch0 epoch1 epoch2 ... epochN (ex: 5 15 30 100 200 300 500 1000)
        dropout = input('dropout').split(' ') # 입력형식: dropout0, dropout1, dropout2, ..., dropoutN (ex: 0 0.05 0.1 0.15 0.2)

        # 딥러닝 학습시키기
        lc = 'mean_squared_error' # loss calculator

        for epo in range(len(epoch)): # 각 epoch에 대하여
            for dro in range(len(dropout)): # 각 dropout에 대하여 실험

                epoc = int(epoch[epo])
                drop = float(dropout[dro])
                
                while True:
                    try:                  
                        # 4. 모델 정의
                        NN = [tf.keras.layers.Reshape((size, size, 1), input_shape=(size*size,)),
                                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(size, size, 1), activation='relu'),
                                keras.layers.MaxPooling2D(pool_size=2),
                                keras.layers.Dropout(drop),
                                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                keras.layers.Flatten(),
                                keras.layers.Dropout(drop),
                                keras.layers.Dense(40, activation='relu'),
                                keras.layers.Dense(size*size, activation='sigmoid'),
                              keras.layers.Reshape((1, size, size), input_shape=(size*size,))]

                        # 5. 옵티마이저
                        if problemNo == 0: op = tf.keras.optimizers.Adam(0.001) # for Sum-
                        else: op = tf.keras.optimizers.Adam(0.0001) # for Common-
                        
                        print('training...')
                        Qlearning_deep_GPU.deepQlearning(0, NN, op, lc, inputs, outputs, None, None, 'WPCNdeepNN', [epoc, None], None, None, False, False, True, deviceName)
                        
                        # 6. 테스트 데이터에 대한 출력 결과 확인하고 정답과 비교하기
                        newModel = Qlearning_deep_GPU.deepLearningModel('WPCNdeepNN_0', False)
                        
                        sumTestThroughput = 0.0 # test throughput의 합계
                        sumCorrectMaxThroughput = 0.0 # 정답의 throughput의 합계 (training data를 생성할 때와 같은 방법으로 최대 throughput 확인)

                        optiInfo = open('optiInfoForMap_' + str(problemNo) + '.txt', 'r')
                        optiInformation = optiInfo.readlines()
                        optiInfo.close()

                        print('testing...')
                        
                        for i in range(numTest):
                            testScreen = originalScreen[numTrain + i] # 테스트할 스크린
                            testOutput = Qlearning_deep_GPU.modelOutput(newModel, [testScreen]) # 테스트 결과
                            testOutputLayer = testOutput[len(testOutput)-1] # 테스트 결과의 output layer의 값

                            # 6-0. 테스트 결과 확인
                            # 출력값 받아오기
                            optiScreen = [[0] * size for j in range(size)] # 테스트 결과의 출력 스크린(=맵)
                            for j in range(size):
                                for k in range(size):
                                    optiScreen[j][k] = Qlearning_deep_GPU_helper.invSigmoid(testOutputLayer[0][0][j][k], size-1)

                            # 출력값에서 최적의 HAP의 x, y좌표 찾기
                            optiY_test = 0 # 출력 map에서의 최적의 HAP의 y좌표
                            optiX_test = 0 # 출력 map에서의 최적의 HAP의 x좌표
                            optiThroughput = 0 # 출력 map에서의 최적의 HAP의 x, y좌표에서의 throughput
                            for j in range(size):
                                for k in range(size):
                                    if optiScreen[j][k] > optiThroughput:
                                        optiY_test = j
                                        optiX_test = k
                                        optiThroughput = optiScreen[j][k]

                            # 상하/좌우의 throughput의 값을 이용하여 좌표 보정
                            optiY = optiY_test
                            optiX = optiX_test

                            # y좌표를 (y-1), y, (y+1) 좌표의 throughput 값을 이용하여 보정
                            if optiY > 0 and optiY < size-1:
                                    
                                up = optiScreen[optiY-1][optiX]
                                this = optiScreen[optiY][optiX]
                                down = optiScreen[optiY+1][optiX]
                                minVal = min(up, this, down)

                                up -= minVal
                                this -= minVal
                                down -= minVal

                                optiY_test += (up * (-1) + down * 1)/(up + this + down)

                            # x좌표를 (x-1), x, (x+1) 좌표의 throughput 값을 이용하여 보정
                            if optiX > 0 and optiX < size-1:

                                left = optiScreen[optiY][optiX-1]
                                this = optiScreen[optiY][optiX]
                                right = optiScreen[optiY][optiX+1]
                                minVal = min(left, this, right)

                                left -= minVal
                                this -= minVal
                                right -= minVal

                                optiX_test += (left * (-1) + right * 1)/(left + this + right)

                            # 출력값 중 HAP의 좌표를 범위 내로 조정
                            if optiY_test > size-1: optiY_test = size-1
                            elif optiY_test < 0: optiY_test = 0
                            
                            if optiX_test > size-1: optiX_test = size-1
                            elif optiX_test < 0: optiX_test = 0

                            # 테스트 결과 받아오기
                            (throughput, HAPtime) = WPCN_helper_REAL.getThroughput(wdList[numTrain + i], [optiY_test, optiX_test], size, problemNo)
                            sumTestThroughput += throughput

                            # 6-1. testScreen과 optiScreen 중 일부 출력
                            if i < 10: # 처음 10개의 map에 대해서만 출력
                                # testScreen 출력
                                print(' --- test input screen for map ' + str(i) + ' ---')
                                toPrint = ''
                                for j in range(size):
                                    toPrint += '['
                                    for k in range(size):
                                        if testScreen[j][k] == -1: toPrint += 'W  '
                                        else: toPrint += '.  '
                                    toPrint += ']\n'
                                print(toPrint)

                                # optiScreen 출력
                                print(' --- test output screen for map ' + str(i) + ' ---')
                                toPrint = ''
                                for j in range(size):
                                    toPrint += '['
                                    for k in range(size): toPrint += str(round(optiScreen[j][k], 3)) + ' '*(7-len(str(round(optiScreen[j][k], 3))))
                                    toPrint += ']\n'
                                print(toPrint)

                            # 6-2. 정답과 비교
                            # 최적의 HAP의 위치 및 할당 시간 찾기
                            # HAP의 위치 및 할당 시간에 따른 throughput의 최댓값
                            correctMaxThroughput = float(optiInformation[(numTrain + i) * (size + 1)].split(' ')[5])
                            sumCorrectMaxThroughput += correctMaxThroughput

                            # 정답과 비교
                            print('test throughput for map ' + str(i) + ' [Y=' + str(round(optiY_test, 3)) + ' X=' + str(round(optiX_test, 3)) + ' HT=' + str(HAPtime) + '] : '
                                  + str(round(throughput, 6)) + ' / ' + str(round(correctMaxThroughput, 6)) + ' ('
                                  + str(round(100.0 * throughput / correctMaxThroughput, 6)) + ' %)')
                            if i < 10: print('')

                        # 7. 평가
                        percentage = round(100.0 * sumTestThroughput / sumCorrectMaxThroughput, 6)
                        print('size:' + str(size) + ' train:' + str(numTrain) + ' test:' + str(numTest)
                              + ' problem(0=Sum,1=Common):' + str(problemNo) + ' epoch:' + str(epoc) + ' dropout:' + str(round(drop, 3)))
                        print('total test throughput: ' + str(round(sumTestThroughput, 6)))
                        print('total max  throughput: ' + str(round(sumCorrectMaxThroughput, 6)))
                        print('percentage           : ' + str(percentage) + ' %')

                        RL.append([size, numTrain, numTest, problemNo, epoc, drop, percentage])

                    # 오류 발생 처리
                    except Exception as ex:
                        print('\n################################')
                        print('error: ' + str(ex))
                        print('retry training and testing')
                        print('################################\n')

                    # 현재까지의 평가 결과 출력
                    print('\n <<< ESTIMATION RESULT >>>')
                    print('No.\tsize\ttrain\ttest\tproblem\tepoch\tdropout\tpercentage')

                    saveResult = '\n <<< ESTIMATION RESULT >>>\nNo.\tsize\ttrain\ttest\tproblem\tepoch\tdropout\tpercentage\n'
                    
                    for i in range(len(RL)):
                        toPrint = (str(i) + '\t' + str(RL[i][0]) + '\t' + str(RL[i][1]) + '\t' + str(RL[i][2]) + '\t'
                              + str(RL[i][3]) + '\t' + str(RL[i][4]) + '\t'
                              + str(round(RL[i][5], 3)) + '\t' + (' ' if round(RL[i][6], 4) < 10.0 else '') + str(round(RL[i][6], 4)))
                        print(toPrint)
                        saveResult += (toPrint + '\n')
                    print('\n')

                    # 평가 결과 저장
                    fSave = open('DL_WPCN_00_result_' + str(problemNo) + '.txt', 'w')
                    fSave.write(saveResult)
                    fSave.close()

                    # 학습 및 테스트 종료
                    break
