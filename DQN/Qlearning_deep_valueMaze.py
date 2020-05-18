import math
import random
import Qlearning_deep
import Qlearning_deep_helper
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

# 이동이 가능한지의 여부 파악
# screen: 게임 스크린
# move  : 이동 (0: 상, 1: 하, 2: 좌, 3: 우)
# noUse : (사용되지 않는 변수)
# point : 캐릭터의 현재 좌표 - point[세로좌표, 가로좌표]
def feasible(screen, move, noUse, point):
    if move == 0 and point[0] > 0: return True # 상
    elif move == 1 and point[0] < len(screen)-1: return True # 하
    elif move == 2 and point[1] > 0: return True # 좌
    elif move == 3 and point[1] < len(screen[1])-1: return True # 우
    return False

# 직접 보상을 구하는 함수
# Q(s, a) <- r + t*max(a')Q(s', a')에서 r을 구하는 함수
def directReward(screen, point):
    return screen[point[0]][point[1]]

# 다음 상태를 구하는 함수
def getNextState(screen, action):
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
            if screen[i][j] == 50: # 현재 캐릭터의 좌표에 해당하는 값 = 50
                point = [i, j]
                broken = True
                break
        if broken == True: break

    # 현재 좌표에서 이동
    screenCopy[point[0]][point[1]] = -50 # 이전 위치의 값을 벽(-50)으로 바꾸기
    if action == 0: point[0] -= 1 # 상
    elif action == 1: point[0] += 1 # 하
    elif action == 2: point[1] -= 1 # 좌
    elif action == 3: point[1] += 1 # 우

    # 이동한 값이 범위를 벗어나면 None을 반환
    if Qlearning_deep_helper.exceed(screenCopy, point[0], point[1]): return None
    
    screenCopy[point[0]][point[1]] = 50

    return screenCopy

# 스크린 표시 함수
# screen 인자는 feasible 함수에서와 같음
def displayScreen(screen, score, count):
    print('< 카운트: ' + str(count) + ' 점수: ' + str(score) + ' >')
    print(' +=' + '====='*len(screen[0]) + '=+')
    for i in range(len(screen)):
        temp = ' | '
        for j in range(len(screen[0])):
            if screen[i][j] == 0: temp += '  .  ' # 빈칸
            elif screen[i][j] == 1: temp += '  +  ' # 플러스 요소
            elif screen[i][j] == 2: temp += ' + + '
            elif screen[i][j] == 3: temp += ' +++ '
            elif screen[i][j] == 4: temp += ' <+> '
            elif screen[i][j] == 5: temp += ' [+] '
            elif screen[i][j] == -1: temp += '  -  ' # 마이너스 요소
            elif screen[i][j] == -2: temp += ' - - '
            elif screen[i][j] == -3: temp += ' --- '
            elif screen[i][j] == -4: temp += ' <-> '
            elif screen[i][j] == -5: temp += ' [-] '
            elif screen[i][j] == 50: temp += '(+_+)' # 캐릭터
            elif screen[i][j] == -50: temp += '#####' # 벽
        print(temp + ' | ')
    print(' +=' + '====='*len(screen[0]) + '=+')
    print('')

# 스크린 초기화
def initScreen(size):
    originalScreen = [[0.0] * size for j in range(size)]
    for j in range(int(size*size/2)):
        centerY = random.randint(0, size-1)
        centerX = random.randint(0, size-1)
        pm = random.randint(0, 8)
        
        if originalScreen[centerY][centerX] != 0.0: continue

        PSF = [[0.2, 0.5, 0.9, 0.5, 0.2],
               [0.5, 1.5, 2.5, 1.5, 0.5],
               [0.9, 2.5, 4.0, 2.5, 0.9],
               [0.5, 1.5, 2.5, 1.5, 0.5],
               [0.2, 0.5, 0.9, 0.5, 0.2]]

        for y in range(centerY-2, centerY+3):
            for x in range(centerX-2, centerX+3):
                if Qlearning_deep_helper.exceed(originalScreen, y, x): continue
                
                if pm >= 4: originalScreen[y][x] += PSF[y-(centerY-2)][x-(centerX-2)]
                else: originalScreen[y][x] -= PSF[y-(centerY-2)][x-(centerX-2)]
                originalScreen[y][x] = round(originalScreen[y][x], 1)

    for j in range(size):
        for k in range(size):
            value = originalScreen[j][k]
            if value >= 4.5: originalScreen[j][k] = 5
            elif value >= 3.5: originalScreen[j][k] = 4
            elif value >= 2.5: originalScreen[j][k] = 3
            elif value >= 1.5: originalScreen[j][k] = 2
            elif value >= 0: originalScreen[j][k] = 1
            elif value > -1.5: originalScreen[j][k] = -1
            elif value > -2.5: originalScreen[j][k] = -2
            elif value > -3.5: originalScreen[j][k] = -3
            elif value > -4.5: originalScreen[j][k] = -4
            else: originalScreen[j][k] = -5

    originalScreen[int((size-1)/2)][int((size-1)/2)] = 0 # 캐릭터가 있는 자리는 빈칸

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
    option = int(input('딥러닝 옵션 입력: 0(기본), 1(DoubleDQN), 2(Duel), 3(경험반복), 4(경험반복&DoubleDQN), 5(경험반복&Duel)'))
    size = int(input('판의 크기 입력'))
    learningSize = int(input('캐릭터를 중심으로 한, 학습할 영역의 크기 입력'))
    
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
    if learningSize < 7: # 일반 Neural Network 적용
        NN = [tf.keras.layers.Flatten(input_shape=(learningSize*learningSize,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(4, activation='sigmoid')]
    else: # CNN 적용
        NN = [tf.keras.layers.Reshape((learningSize, learningSize, 1), input_shape=(learningSize*learningSize,)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(learningSize, learningSize, 1), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=2),
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
        originalScreen = initScreen(size)

        # 초기화된 스크린 불러오기
        screen = Qlearning_deep_helper.arrayCopy(originalScreen)

        # 캐릭터를 중앙에 배치
        point = [int((size-1)/2), int((size-1)/2)]
        screen[point[0]][point[1]] = 50

        # 마지막으로 행동을 실행하기 직전의 상태 (s)
        # 초기값은 스크린과 같음
        lastScreen = Qlearning_deep_helper.arrayCopy(screen)

        # 마지막으로 실행한 행동 (a)
        lastDecision = -1

        # 직전의 좌표
        lastPoint = [int((size-1)/2), int((size-1)/2)]

        count = -1

        ### 1. 게임 진행 ###
        while(1):
            count += 1

            ## 1-0. 캐릭터 이동, 점수 및 스크린 갱신, 표시 ##
            screen[point[0]][point[1]] = -50
            
            if lastDecision == 0: point[0] -= 1 # 상
            elif lastDecision == 1: point[0] += 1 # 하
            elif lastDecision == 2: point[1] -= 1 # 좌
            elif lastDecision == 3: point[1] += 1 # 우

            dR = directReward(lastScreen, point) # direct reward for the action
            score += dR
            
            screen[point[0]][point[1]] = 50
            displayScreen(screen, score, count)

            ## 1-1. 상태 입력, 행동에 따른 보상, 그 행동으로 업데이트된 Q value와 이전 Q value의 차이를 기록 ##
            # ScreenCopy : screen에서 입력으로 넣을 부분 (학습할 부분)
            screenCopy = Qlearning_deep_helper.logData(learningSize=learningSize, charSize=1, screen=screen, point=point,
                                          lastScreen=lastScreen, lastDecision=lastDecision, lastPoint=lastPoint,
                                          dR=dR, states=states, outputs=outputs, Qdiffs=Qdiffs,
                                          exceedValue=0, isFile=isFile, model0=model0, model1=model1, last0=5, last1=10,
                                          func0=0.5, actions=4, count=count, option=option)

            ## 1-2. 게임 오버 처리 ##
            itemLeft = 0 # 남아 있는 아이템 개수
            for j in range(size):
                for k in range(size):
                    if abs(screen[j][k]) == 1 or abs(screen[j][k]) == 2 or abs(screen[j][k]) == 3 or abs(screen[j][k]) == 4 or abs(screen[j][k]) == 5:
                        itemLeft += 1

            # 아이템이 없거나 카운터가 20 이상이면 게임 종료
            if itemLeft == 0 or count >= 20:
                scores.append(score)
                print('최종 점수: ' + str(score))
                break
            
            ## 1-3. 다음 행동 결정 ##
            # 상태 screenCopy(=screen =s)에서 행동 decision(=a)을 결정
            decision = Qlearning_deep_helper.decideAction(option, screen, screenCopy, point, None, isFile,
                                                          model0, model1, last0=5, last1=10, gameNo=i,
                                                          charSize=1, actions=4, exceedValue=0, charValue=50,
                                                          randomProb_=randomProb, feasible_=feasible,
                                                          getNextState_=getNextState)

            ## 1-4. lastScreen, lastDecision, lastPoint 갱신 ##
            for j in range(size):
                for k in range(size): lastScreen[j][k] = screen[j][k]
            lastDecision = decision
            lastPoint = [point[0], point[1]]

        ### 2. 게임 종료 후 딥러닝 실행 ###
        if len(states) < 1000: # 1000개 미만이면 모두 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states, outputs, Qdiffs, 25, 'valueMaze', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        else: # 1000개 이상이면 마지막 1000개만 학습
            Qlearning_deep.deepQlearning(option, NN, op, 'mean_squared_error', states[len(states)-1000:], outputs[len(states)-1000:], Qdiffs[len(states)-1000:], 25, 'valueMaze', [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], False, False, True)
        isFile = True

        if option % 3 == 0 or option % 3 == 1: model0 = Qlearning_deep.deepLearningModel('valueMaze_0', False)
        if option % 3 == 1 or option % 3 == 2: model1 = Qlearning_deep.deepLearningModel('valueMaze_1', False)

        print('\n\n << 점수 목록 >>')
        print(scores)
        print('')
