# 상하좌우 action을 취하면 이동속도의 벡터가 그 방향으로 1 증가함
# 이동벡터의 크기가 5보다 커질수 없음 (가로*가로+세로*세로)
# 캐릭터 크기는 3x3
# 캐릭터가 벽에 닿으면 -5점을 얻으며 게임종료, 공간에만 닿으면 +1점, 연료를 얻으면 +5점, 목표지점(3x3)은 +10점
# 현재 캐릭터가 있는 지점(3x3)은 screen 배열에서 값 50으로 표시
# 52x52 배열, 40x40 이동공간

# 추후 비교 및 함수 helper로 이동을 위해 valueMaze.py로 덮어쓰고 다시 이걸로 덮어쓸것

import math
import random
import Qlearning_deep
import Qlearning_deep_helper
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

# 현재 벡터와 action으로부터 새로운 벡터 구하기
# vector: 이동 속도의 벡터
# action: 이동 (0: 상, 1: 하, 2: 좌, 3: 우)
def getNewVec(vector, action):

    # action을 취한 후의 새로운 이동 벡터
    newVector = [vector[0], vector[1]]
    
    if action == 0: newVector[0] -= 1 # 상
    elif action == 1: newVector[0] += 1 # 하
    elif action == 2: newVector[1] -= 1 # 좌
    elif action == 3: newVector[1] += 1 # 우

    # 새로운 벡터의 크기가 sqrt(5) 이하이면 새로운 벡터, sqrt(5)보다 크면 기존 벡터 반환
    if newVector[0]*newVector[0] + newVector[1]*newVector[1] <= 5: return newVector
    return vector

# 이동이 가능한지의 여부 파악 (배열을 벗어나면 안됨)
# screen: 게임 스크린(42x42 배열)
# move  : 이동 (0: 상, 1: 하, 2: 좌, 3: 우)
# vector: 이동 속도의 벡터
# point : 캐릭터의 현재 좌표 - point[세로좌표, 가로좌표] : 3x3 캐릭터의 중심점 (1, 1) 기준
def feasible(screen, move, vector, point):

    # action을 취한 후의 새로운 이동 벡터
    newVector = getNewVec(vector, move)

    # action을 취해서 변경된 이동 벡터로 이동한 후의 좌표
    newPoint = [point[0], point[1]]
    newPoint[0] += newVector[0]
    newPoint[1] += newVector[1]

    # 배열 범위를 벗어나지 않으면 True를 반환
    if not Qlearning_deep_helper.exceed(screen, newPoint[0], newPoint[1]): return True
    else: return False

# 직접 보상을 구하는 함수
# Q(s, a) <- r + t*max(a')Q(s', a')에서 r을 구하는 함수
def directReward(screen, point):

    # 벽 또는 연료 처리
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            if screen[i][j] == -5: return -5 # 주변 지역이 벽이면 -5를 반환하며 실패
            elif screen[i][j] == 5: return 5 # 주변 지역이 연료이면 +5를 반환

    if screen[point[0]][point[1]] == 50: return 1 # 캐릭터가 있는 지역이면 1을 반환
    
    return screen[point[0]][point[1]]

# screen에서 action을 취한 이후의 다음 상태를 구하는 함수
# screen: 게임 스크린(52x52 배열)
# action: 이동 (0: 상, 1: 하, 2: 좌, 3: 우)
# vector: 이동 속도의 벡터
def getNextState(screen, action, vector):
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
            if screen[i-1][j-1] == 50 and screen[i+1][j+1] == 50: # 현재 캐릭터의 좌표에 해당하는 값 = 50
                point = [i, j] # 캐릭터의 중심점
                broken = True
                break
        if broken == True: break

    # 이전 위치의 값을 벽(-5)으로 바꾸기
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            screenCopy[i][j] = -5

    # 이동한 후의 점의 좌표 구하기
    newVector = getNewVec(vector, action)
    point[0] += newVector[0]
    point[1] += newVector[1]

    # 이동한 값이 범위를 벗어나면 None을 반환
    if Qlearning_deep_helper.exceed(screenCopy, point[0], point[1]): return None
    
    # 새로운 위치의 값을 캐릭터(50)로 바꾸기
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            screenCopy[i][j] = 50

    return screenCopy

# 스크린 표시 함수
# screen 인자는 feasible 함수에서와 같음
# score    : 현재 점수
# point    : 캐릭터의 좌표 (중심점)
# goalPoint: 목표 지점의 좌표 (3x3짜리 목표 지점의 중심점)
# count    : 카운트(턴)
# vector   : 이동 속도의 벡터
def displayScreen(screen, score, point, goalPoint, vector, count):
    print('< 카운트: ' + str(count) + ' 점수: ' + str(score) + ' 이동벡터: ' + str(vector) + ' >')
    print(' +=' + '==='*(len(screen[0])-10) + '=+')
    for i in range(5, len(screen)-5):
        temp = ' | '
        for j in range(5, len(screen[0])-5):

            # 캐릭터
            if abs(i-point[0]) <= 1 and abs(j-point[1]) <= 1:
                if i == point[0]-1 and j == point[1]-1: temp += ' _|'
                elif i == point[0]-1 and j == point[1]: temp += '^o^'
                elif i == point[0]-1 and j == point[1]+1: temp += '|_ '

                elif i == point[0] and j == point[1]-1: temp += '_|_'
                elif i == point[0] and j == point[1]: temp += '___'
                elif i == point[0] and j == point[1]+1: temp += '_|_'

                elif i == point[0]+1 and j == point[1]-1: temp += ' O '
                elif i == point[0]+1 and j == point[1]: temp += '   '
                elif i == point[0]+1 and j == point[1]+1: temp += ' O '

            # 목표 지점
            elif abs(i-goalPoint[0]) <= 1 and abs(j-goalPoint[1]) <= 1:
                if i == goalPoint[0]-1 and j == goalPoint[1]-1: temp += '+--'
                elif i == goalPoint[0]-1 and j == goalPoint[1]: temp += '---'
                elif i == goalPoint[0]-1 and j == goalPoint[1]+1: temp += '--+'

                elif i == goalPoint[0] and j == goalPoint[1]-1: temp += '|G '
                elif i == goalPoint[0] and j == goalPoint[1]: temp += 'O A'
                elif i == goalPoint[0] and j == goalPoint[1]+1: temp += ' L|'

                elif i == goalPoint[0]+1 and j == goalPoint[1]-1: temp += '+--'
                elif i == goalPoint[0]+1 and j == goalPoint[1]: temp += '---'
                elif i == goalPoint[0]+1 and j == goalPoint[1]+1: temp += '--+'

            # 나머지
            else:
                if screen[i][j] == 1: temp += ' . ' # 빈칸
                elif screen[i][j] == -5: temp += '###' # 벽
                elif screen[i][j] == 5: temp += 'OIL' # 벽
        print(temp + ' | ')
    print(' +=' + '==='*(len(screen[0])-10) + '=+')
    print('')

# 스크린 초기화
# 파일 형식: 0(공간 -> +1), 1(벽 -> -5), 2(캐릭터 -> +50), 3(목표지점 -> +10), 4(연료 -> +5)
# point    : 캐릭터의 좌표
# goalPoint: 목표 지점의 좌표
def initScreen(size):
    originalScreen = [[0.0] * size for j in range(size)]
    file = open('vectorCarScreen.txt', 'r')
    read = file.readlines()
    file.close()

    # 각 줄(read[i])마다 오른쪽에 있는 개행문자 제거
    for i in range(size):
        read[i] = read[i].replace('\n', '')
    
    # 캐릭터 및 목표지점 찾기
    for i in range(size):
        for j in range(size):
            if read[i-1][j-1] == '2' and read[i+1][j+1] == '2': point = [i, j]
            elif read[i-1][j-1] == '3' and read[i+1][j+1] == '3': goalPoint = [i, j]

    # originalScreen(초기 스크린) 만들기
    for i in range(size):
        for j in range(size):
            if read[i][j] == '0': originalScreen[i][j] = 1
            elif read[i][j] == '1': originalScreen[i][j] = -5
            elif read[i][j] == '2': originalScreen[i][j] = 50
            elif read[i][j] == '3': originalScreen[i][j] = 10
            elif read[i][j] == '4': originalScreen[i][j] = 5

    assert(size == len(read) and size == len(read[0]))
    return [originalScreen, point, goalPoint]

# random prob (랜덤하게 action을 실행할 확률)
# Double DQN의 경우 다음상태에서의 행동의 보상의 최댓값을 이용하므로 현재상태의 값이 안좋아도 계속 높은 평가값 유지 가능
# 따라서 이때는 먼저 랜덤으로 10회 학습
def randomProb(i, option):
    if i < 10 and option % 3 == 1: return 1.0
    return 0.0

# 게임 진행
if __name__ == '__main__':

    # option % 3 == 0 -> 기본 Neural Network (_0: reward)
    # option % 3 == 1 -> Double DQN (_0: reward, _1: action that returns max reward)
    # option % 3 == 2 -> Dueling Network (_1: Q(s,a) = V(s)+A(s, a))
    option = int(input('딥러닝 옵션 입력: 0(기본), 1(DoubleDQN), 2(Duel), 3(경험반복), 4(경험반복&DoubleDQN), 5(경험반복&Duel)'))
    size = 52
    learningSize = int(input('캐릭터를 중심으로 한, 학습할 영역의 크기 입력(11 이상 권장)'))
    
    states = [] # 딥러닝 입력값 저장
    outputs = [] # 딥러닝 출력값 저장
    nextStates = [] # 다음 상태 (각 action에 대한 다음 상태)
    output = [0.5, 0.5, 0.5, 0.5] # 현재 state에서의 딥러닝 계산 결과 각 행동에 대한 보상
    nextstt = [[], [], [], []] # 현재 state에서의 딥러닝 결과 각 행동에 의한 다음 상태
    for i in range(4):
        nextstt[i] = [0.0]*(learningSize*learningSize)
    
    Qdiffs = [] # 해당되는 Q value 갱신으로 Q value가 얼마나 달라졌는가?
    
    isFile = False # valueMaze_X.json 학습 모델 파일이 있는가?

    # 모델
    NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(4, activation='sigmoid')]
    op = tf.keras.optimizers.Adam(0.001)

    scores = [] # 점수
    games = 200 # 총 게임 횟수

    model0 = None # 1번째 Neural Network
    model1 = None # 2번째 Neural Network

    for i in range(games):
        print('\n\n <<<< GAME ' + str(i) + ' >>>>\n\n')

        score = 0

        ### 0. 스크린 초기화 ###
        initScreen_ = initScreen(size)
        originalScreen = initScreen_[0]
        point = initScreen_[1]
        goalPoint = initScreen_[2]

        # 초기화된 스크린 불러오기
        screen = Qlearning_deep_helper.arrayCopy(originalScreen)

        # 마지막으로 행동을 실행하기 직전의 상태 (s)
        # 초기값은 스크린과 같음
        lastScreen = Qlearning_deep_helper.arrayCopy(screen)
        
        vector = [0, 0] # 이동 벡터

        # 마지막으로 실행한 행동 (a)
        lastDecision = -1

        # 직전의 캐릭터의 좌표
        lastPoint = [point[0], point[1]]

        count = -1

        ### 1. 게임 진행 ###
        while(1):
            count += 1

            ## 1-0. 캐릭터 이동, 점수 및 스크린 갱신, 표시 ##
            
            # 이전 위치의 값을 벽(-5)으로 바꾸기
            for j in range(point[0]-1, point[0]+2):
                for k in range(point[1]-1, point[1]+2):
                    screen[j][k] = -5

            # 이동 벡터 변경 및 캐릭터 이동
            newVector = getNewVec(vector, lastDecision)
            point[0] += newVector[0]
            point[1] += newVector[1]
            vector = [newVector[0], newVector[1]]

            dR = directReward(lastScreen, point) # direct reward for the action
            score += dR
            
            # 이동한 후의 새로운 위치의 값을 캐릭터(50)로 바꾸기
            for j in range(point[0]-1, point[0]+2):
                for k in range(point[1]-1, point[1]+2):
                    screen[j][k] = 50
            
            displayScreen(screen, score, point, goalPoint, vector, count)

            ## 1-1. 상태 입력, 행동에 따른 보상, 그 행동으로 업데이트된 Q value와 이전 Q value의 차이를 기록 ##
            # ScreenCopy : screen에서 입력으로 넣을 부분 (학습할 부분)
            screenCopy = Qlearning_deep_helper.logData(learningSize=learningSize, charSize=3, screen=screen, point=point,
                                          lastScreen=lastScreen, lastDecision=lastDecision, lastPoint=lastPoint,
                                          dR=dR, states=states, outputs=outputs, Qdiffs=Qdiffs,
                                          exceedValue=-5, isFile=isFile, model0=model0, model1=model1, last0=6, last1=11,
                                          func0=0.5, actions=4, count=count, option=option)

            ## 1-2. 게임 오버 처리 ##
            # 캐릭터의 중심이 목표 지점 9칸 중 1칸에 도달하거나, 캐릭터 범위(3x3) 중 벽이 있으면 종료
            if (abs(point[0]-goalPoint[0]) <= 1 and abs(point[1]-goalPoint[1]) <= 1) or dR == -5:
                scores.append(score)
                print('최종 점수: ' + str(score))
                break
            
            ## 1-3. 다음 행동 결정 ##
            # 상태 screenCopy(=screen =s)에서 행동 decision(=a)을 결정
            decision = Qlearning_deep_helper.decideAction(option, screen, screenCopy, point, vector, isFile,
                                                          model0, model1, last0=6, last1=11, gameNo=i,
                                                          charSize=3, actions=4, exceedValue=-5, charValue=50,
                                                          randomProb_=randomProb, feasible_=feasible,
                                                          getNextState_=getNextState)

            ## 1-4. lastScreen, lastDecision, lastPoint 갱신 ##
            for j in range(size):
                for k in range(size): lastScreen[j][k] = screen[j][k]
            lastDecision = decision
            lastPoint = [point[0], point[1]]

        ### 2. 게임 종료 후 딥러닝 실행 ###
        if len(states) < 1000: # 1000개 미만이면 모두 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states, outputs, Qdiffs, 25, 'vectorCar', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        else: # 1000개 이상이면 마지막 1000개만 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states[len(states)-1000:], outputs[len(states)-1000:], Qdiffs[len(states)-1000:], 25, 'vectorCar', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        isFile = True

        if option % 3 == 0 or option % 3 == 1: model0 = Qlearning_deep.deepLearningModel('vectorCar_0', False)
        if option % 3 == 1 or option % 3 == 2: model1 = Qlearning_deep.deepLearningModel('vectorCar_1', False)

        print('\n\n << 점수 목록 >>')
        print(scores)
        print('')
