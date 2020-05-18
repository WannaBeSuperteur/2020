# 목적:
# WPCN에서 HAP(=drone)의 위치와 충전 시간 할당 방법을 최적화한다.

# 입력: 현재 state를 중심으로 한 odd*odd 크기의 영역
# 입력의 state는 맵으로 하며, 맵은 map.txt의 config에 따라 설정한다.
#
# map.txt의 구성:
# (맵 세로길이) (맵 가로길이) (wireless device의 개수)

# 출력: 각 action별 reward
# action은 HAP(=drone)의 (4방향으로의 1칸씩 움직임 또는 정지) * (충전시간 배정방법)
#     - 충전시간 배정방법: 총 충전시간이 1, HAP의 충전시간이 (1-x)라고 하면 각 device의 충전시간은 가까운 것부터
#       x(1-x), x^2(1-x), x^3(1-x), ...로 지정한다. (즉, (1-x) + x(1-x) + x^2(1-x) + ... = (1-x)/(1-x) = 1)
#     - SUM-THROUGHPUT MAXIMIZATION에서 1-x = 0.1, 0.5, 0.9 (x = 0.9, 0.5, 0.1)
#     - COMMON THROUGHPUT MAXIMIZATION에서 1-x = 0.05, 0.15, 0.3 (x = 0.95, 0.85, 0.7)
#     - 따라서 출력은 5 * 3 = 15가지 존재
#
# <SUM-THROUGHPUT MAXIMIZATION에서>
# reward는 각 WD의 throughput의 합으로 계산한다.
#     - 각 WD의 throughput 계산법: [해당 wireless device에 할당된 시간] * (맵의 크기)^2 / (HAP과 해당 WD 간의 거리)^2
#     - 거리가 가까운 순서대로 시간을 많이 할당
#
# <COMMON THROUGHPUT MAXIMIZATION에서>
# reward는 common throughput과 같다.
# 여기서 (Common Throughput) R_ = max(R_, t) R_ s.t. Ri(t) >= R_ 이므로 각 WD의 throughput의 최솟값과 같다.
#     - 각 WD의 throughput 계산법: [해당 wireless device에 할당된 시간] * (맵의 크기)^2 / (HAP과 해당 WD 간의 거리)^2
#     - 거리가 먼 순서대로 시간을 많이 할당

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

# 움직임의 방향(이동 벡터) 구하기
# move  : 이동 (4방향 및 정지 * 3가지 충전시간)
def getDirection(move):

    # 움직임의 방향 추출
    direction = math.floor(move / 3)
    if direction == 0: direc = [-1, 0] # 상
    elif direction == 1: direc = [1, 0] # 하
    elif direction == 2: direc = [0, -1] # 좌
    elif direction == 3: direc = [0, 1] # 우
    else: direc = [0, 0] # 정지 (direction == 8 일 때)

    return direc

# 이동이 가능한지의 여부 파악 (배열을 벗어나면 안됨)
# screen: 스크린 (odd*odd size)
# move  : 이동 (4방향 및 정지 * 3가지 충전시간)
# point : 드론의 현재 좌표
def feasible(screen, move, noUse, point):

    direc = getDirection(move)

    # direction을 따라 이동한 결과의 함수
    newPoint = [point[0], point[1]]
    newPoint[0] += direc[0]
    newPoint[1] += direc[1]

    # 배열 범위를 벗어나지 않고 wireless device가 있는 곳도 아니면 True를 반환
    if Qlearning_deep_helper.exceed(screen, newPoint[0], newPoint[1]): return False
    if screen[newPoint[0]][newPoint[1]] == -1: return False
    else: return True

# 직접 보상을 구하는 함수
# Q(s, a) <- r + t*max(a')Q(s', a')에서 r을 구하는 함수
# wdList    : wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]
# point     : 드론의 현재 좌표
# screenSize: screen의 크기 (= 가로길이 = 세로길이)
# HAPtime   : 전체 충전 시간(=1)에 대한 드론의 충전 시간의 비율(=1-x)
# problemNo : 0이면 sum throughtput maximization, 1이면 common throughtput maximization
def directReward(wdList, point, screenSize, HAPtime, problemNo):
    return WPCN_helper.getThroughput(wdList, point, screenSize, HAPtime, problemNo)

# screen에서 action을 취한 이후의 다음 상태를 구하는 함수
# screen: 스크린
# action: 이동 (4방향 및 정지 * 3가지 충전시간)
def getNextState(screen, action, noUse):
    point = [-1, -1] # 현재 좌표

    # 현재 스크린 복사 (학습할 부분이 아니라 단순히 복사임)
    screenCopy = []
    for i in range(len(screen)):
        temp = []
        for j in range(len(screen[0])):
            temp.append(screen[i][j])
        screenCopy.append(temp)
    
    # 현재 좌표 찾기
    for i in range(len(screen)):
        broken = False
        for j in range(len(screen[0])):
            if screen[i-1][j-1] == 50 and screen[i+1][j+1] == 1: # 현재 HAP의 좌표에 해당하는 값 = 1
                point = [i, j] # HAP의 중심점
                broken = True
                break
        if broken == True: break

    # 이동한 후의 점의 좌표 구하기
    direc = getDirection(action)
    point[0] += direc[0]
    point[1] += direc[1]

    # 이동한 값이 범위를 벗어나면 None을 반환
    if Qlearning_deep_helper.exceed(screenCopy, point[0], point[1]): return None
    
    # 새로운 위치의 값을 HAP(1)으로 바꾸기
    screenCopy[point[0]][point[1]] = 1

    return screenCopy

# random prob (랜덤하게 action을 실행할 확률)
# Double DQN의 경우 다음상태에서의 행동의 보상의 최댓값을 이용하므로 현재상태의 값이 안좋아도 계속 높은 평가값 유지 가능
# 따라서 이때는 먼저 랜덤으로 10회 학습
def randomProb(i, option):
    if i < 10: return 1.0
    return 0.0

# 진행
if __name__ == '__main__':

    # option % 3 == 0 -> 기본 Neural Network (_0: reward)
    # option % 3 == 1 -> Double DQN (_0: reward, _1: action that returns max reward)
    # option % 3 == 2 -> Dueling Network (_1: Q(s,a) = V(s)+A(s, a))

    moveCases = 5 # 움직임의 경우의 수
    chargeTimeCases = 3 # 충전 시간 결정의 경우의 수
    cases = moveCases * chargeTimeCases # 움직임, 충전 시간 결정의 전체 경우의 수
    
    option = int(input('딥러닝 옵션 입력: 0(기본), 1(DoubleDQN - 사용불가), 2(Duel), 3(경험반복), 4(경험반복&DoubleDQN - 사용불가), 5(경험반복&Duel)'))
    assert(option % 3 != 1)
    learningSize = int(input('HAP을 중심으로 한, 학습할 영역의 크기 입력'))
    problemNo = int(input('0->sum throughtput maximization, 1->common throughtput maximization'))

    # 파일을 읽어서 size 값 구하기
    # 파일 형식 : (맵 세로길이) (맵 가로길이) (wireless device의 개수)
    file = open('map.txt', 'r')
    read = file.readlines()
    readSplit = read[0].split(' ')
    height = int(readSplit[0]) # 맵의 세로길이 (= 가로길이 = size)
    size = height
    file.close()
    
    states = [] # 딥러닝 입력값 저장
    outputs = [] # 딥러닝 출력값 저장
    nextStates = [] # 다음 상태 (각 action에 대한 다음 상태)
    output = [0.5] * cases # 현재 state에서의 딥러닝 계산 결과 각 행동에 대한 보상
    nextstt = [[]] * cases # 현재 state에서의 딥러닝 결과 각 행동에 의한 다음 상태
    for i in range(cases):
        nextstt[i] = [0.0]*(learningSize*learningSize)
    
    Qdiffs = [] # 해당되는 Q value 갱신으로 Q value가 얼마나 달라졌는가?
    
    isFile = False # valueMaze_X.json 학습 모델 파일이 있는가?

    # 모델
    if learningSize >= 10: # 학습 영역의 크기가 10 이상
        NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(40, activation='relu'),
                keras.layers.Dense(cases, activation='sigmoid')]
        last0 = 6
        last1 = 11
    elif learningSize >= 6: # 학습 영역의 크기가 6 이상
        NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Flatten(),
                keras.layers.Dense(40, activation='relu'),
                keras.layers.Dense(cases, activation='sigmoid')]
        last0 = 5
        last1 = 10
    else: # 학습 영역의 크기가 6 미만
        NN = [tf.keras.layers.Flatten(input_shape=(learningSize*learningSize,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(cases, activation='sigmoid')]
        last0 = 5
        last1 = 10

    # 옵티마이저
    op = tf.keras.optimizers.Adam(0.001)

    scores = [] # 점수
    games = 200 # 총 진행 횟수

    model0 = None # 1번째 Neural Network
    model1 = None # 2번째 Neural Network

    for i in range(games):
        print('\n\n <<<< ROUND ' + str(i) + ' >>>>\n\n')

        score = 0

        ### 0. 스크린 초기화 ###
        initScreen_ = WPCN_helper.initScreen(size, True)
        originalScreen = initScreen_[0]
        point = initScreen_[1]
        wdList = initScreen_[2]

        # 초기화된 스크린 불러오기
        screen = Qlearning_deep_helper.arrayCopy(originalScreen)

        # 마지막으로 행동을 실행하기 직전의 상태 (s)
        # 초기값은 스크린과 같음
        lastScreen = Qlearning_deep_helper.arrayCopy(screen)

        # 마지막으로 실행한 행동 (a)
        lastDecision = -1

        # 직전의 캐릭터의 좌표
        lastPoint = [point[0], point[1]]

        count = -1

        ### 1. 진행 ###
        while(1):
            count += 1

            ## 1-0. 캐릭터 이동, 점수 및 스크린 갱신, 표시 ##
            
            # 이전 위치의 값을 공간(0)으로 바꾸기
            screen[point[0]][point[1]] = 0

            # 캐릭터 이동
            direc = getDirection(lastDecision)
            point[0] += direc[0]
            point[1] += direc[1]

            # HAP의 충전 시간(1-x) 구하기
            # SUM-THROUGHPUT MAXIMIZATION에서 1-x = 0.1, 0.5, 0.9 (x = 0.9, 0.5, 0.1)
            # COMMON THROUGHPUT MAXIMIZATION에서 1-x = 0.05, 0.15, 0.3 (x = 0.95, 0.85, 0.7)
            if problemNo == 0: # Sum-throughput maximization (throughput의 합)
                if lastDecision % 3 == 0: HAPtime = 0.1
                elif lastDecision % 3 == 1: HAPtime = 0.5
                elif lastDecision % 3 == 2: HAPtime = 0.9
            elif problemNo == 1: # Common throughput maximization problem (throughput의 최솟값)
                if lastDecision % 3 == 0: HAPtime = 0.05
                elif lastDecision % 3 == 1: HAPtime = 0.15
                elif lastDecision % 3 == 2: HAPtime = 0.3

            dR = directReward(wdList, point, size, HAPtime, problemNo) # direct reward for the action
            score = dR
            if count == 0: lastDR = dR # 직전의 direct reward 값 초기화
            
            # 이동한 후의 새로운 위치의 값을 HAP(1)으로 바꾸기
            screen[point[0]][point[1]] = 1
            
            WPCN_helper.displayScreen(screen, score, count, HAPtime)

            ## 1-1. 상태 입력, 행동에 따른 보상, 그 행동으로 업데이트된 Q value와 이전 Q value의 차이를 기록 ##
            # ScreenCopy : screen에서 입력으로 넣을 부분 (학습할 부분)
            # 단, direct reward는 직전의 direct reward와 비교한 값을 이용: log2(dR/lastDR)
            screenCopy = Qlearning_deep_helper.logData(learningSize=learningSize, charSize=1, screen=screen, point=point,
                                          lastScreen=lastScreen, lastDecision=lastDecision, lastPoint=lastPoint,
                                          dR=math.log(dR/lastDR, 2.0), states=states, outputs=outputs, Qdiffs=Qdiffs,
                                          exceedValue=-1, isFile=isFile, model0=model0, model1=model1, last0=last0, last1=last1,
                                          func0=0.5, actions=cases, count=count, option=option)

            ## 1-2. 종료 처리 ##
            # count의 값이 16 이상이 되면 종료
            if count >= 16:
                scores.append(score)
                print('최종 점수: ' + str(score))
                break
            
            ## 1-3. 다음 행동 결정 ##
            # 상태 screenCopy(=screen =s)에서 행동 decision(=a)을 결정
            decision = Qlearning_deep_helper.decideAction(option, screen, screenCopy, point, None, isFile,
                                                          model0, model1, last0=last0, last1=last1, gameNo=i,
                                                          charSize=1, actions=cases, exceedValue=-1, charValue=1,
                                                          randomProb_=randomProb, feasible_=feasible,
                                                          getNextState_=getNextState)

            ## 1-4. lastScreen, lastDecision, lastPoint, lastDR 갱신 ##
            for j in range(size):
                for k in range(size): lastScreen[j][k] = screen[j][k]
            lastDecision = decision
            lastPoint = [point[0], point[1]]
            lastDR = dR

        ### 2. 종료 후 딥러닝 실행 ###
        if len(states) < 1000: # 1000개 미만이면 모두 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states, outputs, Qdiffs, 25, 'WPCN', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        else: # 1000개 이상이면 마지막 1000개만 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states[len(states)-1000:], outputs[len(states)-1000:], Qdiffs[len(states)-1000:], 25, 'WPCN', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        isFile = True

        if option % 3 == 0 or option % 3 == 1: model0 = Qlearning_deep.deepLearningModel('WPCN_0', False)
        if option % 3 == 1 or option % 3 == 2: model1 = Qlearning_deep.deepLearningModel('WPCN_1', False)

        print('\n\n << 점수 목록 >>')
        print(scores)
        print('')
