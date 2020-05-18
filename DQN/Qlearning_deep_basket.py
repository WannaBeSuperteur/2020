import math
import random
import Qlearning_deep
import Qlearning_deep_helper
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

# 이동이 가능한지의 여부 파악
# screen: 게임 스크린
# move  : 이동 (0: 좌, 1: 정지, 2: 우)
# noUse : (사용되지 않는 변수)
# point : 캐릭터의 현재 좌표 - point[세로좌표, 가로좌표]
def feasible(screen, move, noUse, point):
    if move == 0 and point[1] > 0: return True # 좌
    elif move == 1: return True # 정지
    elif move == 2 and point[1] < len(screen[0])-1: return True # 우
    return False

# 직접 보상을 구하는 함수
# Q(s, a) <- r + t*max(a')Q(s', a')에서 r을 구하는 함수
def directReward(screen, point):
    return screen[point[0]][point[1]]

# 스크린 표시 함수
# 빈칸=0, 캐릭터머리=50, 캐릭터몸=-1, 점수=+5, 벽=-1
# screen 인자는 feasible 함수에서와 같음
def displayScreen(screen, score, count, itemTotalScore):
    print('< 카운트: ' + str(count) + ' 점수: ' + str(score) + ' / ' + str(itemTotalScore) + ' >')
    print(' +=' + '====='*len(screen[0]) + '=+')
    for i in range(len(screen)):
        temp = ' | '
        for j in range(len(screen[0])):
            if screen[i][j] == 0:
                if i == len(screen)-2: temp += '-----' # 캐릭터가 움직이는 구간
                else: temp += '  .  ' # 빈칸
            elif screen[i][j] == 1: temp += '  +  ' # 플러스 요소
            elif screen[i][j] == 3: temp += ' +++ ' # 플러스 3점 요소(속도 2배)
            elif screen[i][j] == -5: temp += ' (#) ' # 폭탄
            elif screen[i][j] == 50: temp += '[+_+]' # 캐릭터
        print(temp + ' | ')
    print(' +=' + '====='*len(screen[0]) + '=+')
    print('')

# 스크린 초기화
def initScreen(size):
    originalScreen = [[0.0] * size for j in range(size)]

    originalScreen[size-2][int((size-1)/2)] = 0 # 캐릭터가 있는 자리는 빈칸

    return originalScreen

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
    option = int(input('딥러닝 옵션 입력: 0(기본), 1(DoubleDQN - 사용불가), 2(Duel), 3(경험반복), 4(경험반복&DoubleDQN - 사용불가), 5(경험반복&Duel)'))
    assert(option % 3 != 1)
    size = int(input('판의 크기 입력'))
    learningSize = int(input('캐릭터를 중심으로 한, 학습할 영역의 크기 입력'))

    assert(learningSize % 2 == 1)
    
    states = [] # 딥러닝 입력값 저장
    outputs = [] # 딥러닝 출력값 저장
    nextStates = [] # 다음 상태 (각 action에 대한 다음 상태)
    output = [0.6, 0.6, 0.6] # 현재 state에서의 딥러닝 계산 결과 각 행동에 대한 보상
    nextstt = [[], [], []] # 현재 state에서의 딥러닝 결과 각 행동에 의한 다음 상태
    for i in range(3):
        nextstt[i] = [0.0]*(learningSize*learningSize)
    
    Qdiffs = [] # 해당되는 Q value 갱신으로 Q value가 얼마나 달라졌는가?
    
    isFile = False # valueMaze_X.json 학습 모델 파일이 있는가?

    # 모델
    if learningSize >= 10:
        NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(40, activation='relu'),
                keras.layers.Dense(3, activation='sigmoid')]
        last0 = 6
        last1 = 11
    elif learningSize >= 6:
        NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Flatten(),
                keras.layers.Dense(40, activation='relu'),
                keras.layers.Dense(3, activation='sigmoid')]
        last0 = 5
        last1 = 10
    else:
        NN = [tf.keras.layers.Flatten(input_shape=(learningSize*learningSize,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='sigmoid')]
        last0 = 5
        last1 = 10
        
    op = tf.keras.optimizers.Adam(0.001)

    scores = [] # 점수
    games = 200 # 총 게임 횟수

    model0 = None # 1번째 Neural Network
    model1 = None # 2번째 Neural Network

    for i in range(games):
        print('\n\n <<<< GAME ' + str(i) + ' >>>>\n\n')

        score = 0
        center = int((size-1)/2) # 중앙의 좌표

        ### 0. 스크린 초기화 ###
        originalScreen = initScreen(size)

        # 초기화된 스크린 불러오기
        screen = Qlearning_deep_helper.arrayCopy(originalScreen)

        # 캐릭터를 라인의 중앙에 배치
        point = [size-2, center]
        screen[point[0]][point[1]] = 50

        # 마지막으로 행동을 실행하기 직전의 상태 (s)
        # 초기값은 스크린과 같음
        lastScreen = Qlearning_deep_helper.arrayCopy(screen)

        # 마지막으로 실행한 행동 (a)
        lastDecision = -1

        # 직전의 좌표 (처음에는 캐릭터 좌표의 초기값으로 초기화)
        lastPoint = [size-2, center]

        count = -1
        itemTotalScore = 0 # 얻을 수 있는 점수 아이템의 총점

        ### 1. 게임 진행 ###
        while(1):
            count += 1

            ## 1-0. 캐릭터 이동, 점수 및 스크린 갱신, 표시 ##
            screen[point[0]][point[1]] = 0
            oldPoint = [point[0], point[1]]
            
            if lastDecision == 1: point[1] += 0 # 정지
            elif lastDecision == 0: point[1] -= 1 # 좌
            elif lastDecision == 2: point[1] += 1 # 우

            # 점수 아이템 이동을 위한 체크 배열
            checked = [[False] * size for j in range(size)]

            # 점수 아이템 이동
            for j in range(size):
                for k in range(size):
                    if checked[j][k]: continue
                    
                    if screen[j][k] == 1:
                        screen[j][k] = 0
                        if j < size-1:
                            screen[j+1][k] = 1
                            checked[j+1][k] = True
                    elif screen[j][k] == 3 or screen[j][k] == -5:
                        temp = screen[j][k]
                        screen[j][k] = 0
                        if j < size-3:
                            screen[j+2][k] = temp
                            checked[j+2][k] = True
                        elif j < size-1:
                            screen[j+1][k] = temp
                            checked[j+1][k] = True

            # 랜덤한 위치(빈칸)에 점수 아이템 추가
            for j in range(size):
                noItem = True # j열에 아이템이 없으면 True
                for k in range(size):
                    if screen[k][j] == 1 or screen[k][j] == 3 or screen[k][j] == -5:
                        noItem = False

                itemThrowOrNot = random.random() # 점수 아이템 추가 확률(0.15 + count * 0.002, 최대 0.5)보다 작으면 아이템 추가
                itemType = random.randint(0, 2) # 0=1점, 1=3점, 2=폭탄(-5점)
                itemY = random.randint(0, 1) # 아이템의 초기 y좌표 (0 또는 1)
                
                if noItem and itemThrowOrNot < min(0.15 + count * 0.002, 0.5):
                    if itemType == 0: screen[itemY][j] = 1
                    elif itemType == 1: screen[itemY][j] = 3
                    elif itemType == 2: screen[itemY][j] = -5

            # 직접 보상 계산 (캐릭터가 배치되지 않고 점수 아이템이 이동한 후의 상태)
            dR = directReward(screen, point) # direct reward for the action
            if count > 0: score += dR

            # 맨 밑줄의 아이템 총점 계산 #
            for j in range(size):
                if screen[size-1][j] > 0: itemTotalScore += screen[size-1][j]

            # 캐릭터 재배치 후 스크린 표시
            screen[point[0]][point[1]] = 50
            displayScreen(screen, score, count, itemTotalScore)

            ## 1-1. 상태 입력, 행동에 따른 보상, 그 행동으로 업데이트된 Q value와 이전 Q value의 차이를 기록 ##
            # ScreenCopy : screen에서 입력으로 넣을 부분 (학습할 부분)
            # 배열 index 범위 초과는 -1점 처리
            screenCopy = Qlearning_deep_helper.logData(learningSize=learningSize, charSize=1, screen=screen, point=point,
                                          lastScreen=lastScreen, lastDecision=lastDecision, lastPoint=lastPoint,
                                          dR=dR, states=states, outputs=outputs, Qdiffs=Qdiffs,
                                          exceedValue=-1, isFile=isFile, model0=model0, model1=model1, last0=last0, last1=last1,
                                          func0=0.6, actions=3, count=count, option=option)

            ## 1-2. 게임 오버 처리 ##
            # 지금까지 얻을 수 있었던 점수 아이템의 총점이 100점이 되면 게임 오버 #
            if itemTotalScore >= 100:
                scores.append(score)
                print('최종 점수: ' + str(score))
                break
            
            ## 1-3. 다음 행동 결정 ##
            # 상태 screenCopy(=screen =s)에서 행동 decision(=a)을 결정
            decision = Qlearning_deep_helper.decideAction(option, screen, screenCopy, point, None, isFile,
                                                          model0, model1, last0=last0, last1=last1, gameNo=i,
                                                          charSize=1, actions=3, exceedValue=-1, charValue=50,
                                                          randomProb_=randomProb, feasible_=feasible,
                                                          getNextState_=None)

            ## 1-4. lastScreen, lastDecision, lastPoint 갱신 ##
            for j in range(size):
                for k in range(size): lastScreen[j][k] = screen[j][k]
            lastDecision = decision
            lastPoint = [point[0], point[1]]

        ### 2. 게임 종료 후 딥러닝 실행 ###
        if len(states) < 1000: # 1000개 미만이면 모두 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states, outputs, Qdiffs, 25, 'basket', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        else: # 1000개 이상이면 마지막 1000개만 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states[len(states)-1000:], outputs[len(states)-1000:], Qdiffs[len(states)-1000:], 25, 'basket', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        isFile = True

        if option % 3 == 0 or option % 3 == 1: model0 = Qlearning_deep.deepLearningModel('basket_0', False)
        if option % 3 == 1 or option % 3 == 2: model1 = Qlearning_deep.deepLearningModel('basket_1', False)

        print('\n\n << 점수 목록 >>')
        print(scores)
        print('')
