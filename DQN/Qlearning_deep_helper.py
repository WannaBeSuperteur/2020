import math
import random
import Qlearning_deep
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

import warnings
warnings.filterwarnings("error")

# Flatten
def flatten(array):
    result = []
    for i in range(len(array)):
        for j in range(len(array[0])): result.append(array[i][j])
    return result

# sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# inverse of sigmoid
# y = 1/(1+e^(-x))
# x = 1/(1+e^(-y))
# 1/x = 1+e^(-y)
# 1/x - 1 = e^(-y)
# ln(1/x - 1) = -y
# ln((1-x)/x) = -y
# ln(x/(1-x)) = y -> y = ln(x/(1-x))
def invSigmoid(x, highBound):
    try:
        preResult = x / (1 - x)
        result = math.log(preResult)
        return result
    except RuntimeWarning: # catch runtime warnings
        print('value of x is ' + str(x) + ' -> highBound = ' + str(highBound))
        if x < 0.5: return 0.0 # return value must be >= 0.0
        else: return highBound

# exceed
def exceed(array, i, j):
    if i < 0 or j < 0 or i >= len(array) or j >= len(array[0]): return True
    return False

# 지금까지의 (state(=screen) -> action별 reward) 데이터를 학습한 모델에 state를 입력하여 나온 출력을 근거로
# 현재 캐릭터 좌표에서 각 이동 방향(action)에 따른 Q value 중 최댓값을 반환
# screen의 값으로 새로운 상태 s'를 이용하는 경우 Q(s, a) <- r + 0.9*max(a')Q(s', a')에서 max(a')Q(s', a')를 구하는 함수

# screen       : 게임 스크린
# isFile       : valueMaze_X.json 학습 모델 파일이 있는가?
# model0,model1: 데이터를 학습한 모델
# last0,last1  : model0(기본 DQN, Double DQN), model1(Dueling DQN)의 마지막 layer의 index
# option       : 딥러닝 옵션
def maxQvalue(screen, isFile, model0, model1, last0, last1, option):

    # 새로운 상태(=s')에서 각 행동 a'를 취했을 때의 보상값 Q(s', a') 구하기
    if isFile:
        if option % 3 == 0 or option % 3 == 1: Qvalues = Qlearning_deep.modelOutput(model0, [screen])
        elif option % 3 == 2: Qvalues = Qlearning_deep.modelOutput(model1, [screen])
    else: return 0

    # 그 보상값의 최댓값 반환
    if option % 3 == 2: return max(Qvalues[last1][0])
    return max(Qvalues[last0][0])

# 현재 상태에서 각 행동을 취했을 때의 Q value의 배열을 반환하는 함수

# screen       : 게임 스크린
# isFile       : valueMaze_X.json 학습 모델 파일이 있는가?
# model0,model1: 데이터를 학습한 모델
# last0,last1  : model0(기본 DQN, Double DQN), model1(Dueling DQN)의 마지막 layer의 index
# func0        : 활성화 함수에 0을 넣었을 때의 값 (예: sigmoid(0)=0.5)
# actions      : 행동의 개수
# option       : 딥러닝 옵션
def getQvalues(screen, isFile, model0, model1, last0, last1, func0, actions, option):

    # 각 action에 대해 Q(s, a)의 값 구하기
    if isFile:
        if option % 3 == 0 or option % 3 == 1:
            Qvalues = Qlearning_deep.modelOutput(model0, [screen]) # 이미 flatten된 screen임
            if option % 3 == 0: print('Qval of decision whose next state is above:\n' + str(Qvalues[last0][0]) + '\n\n')
            return Qvalues[last0][0]
        elif option % 3 == 2:
            Qvalues = Qlearning_deep.modelOutput(model1, [screen])
            print('Qval of decision whose next state is above:\n' + str(Qvalues[last1][0]) + '\n\n')
            return Qvalues[last1][0]
        
    else: return [func0]*actions

# 행동 결정 함수
# Q(s, a), 즉 Q(s_t, a_t) 값이 가장 큰 행동 a를 선택

# option == 0 (기본) -> Q(s_t, a_t) = r + 0.9 * max(a)Q(s_(t+1), a)
# option == 1 (Double DQN) -> Q(s_t, a_t) = r + 0.9 * Q(s_(t+1), argmax(a)Q(s_(t+1), a))
#                          -> argmax(a)Q(s_(t+1), a) 부분은 model 1, Q(s_(t+1), ...) 부분은 model 0이 담당
# option == 2 (Dueling DQN) -> Q(s_t, a_t) = V(s_t) + A(s_t, a_t)

# screen       : 게임 스크린
# screenCopy   : screen에서 학습할 부분만 복사한 것

# point        : 캐릭터의 현재 좌표 - point[세로좌표, 가로좌표]
# additional   : 추가적인 정보(이동 속도의 벡터 등)

# isFile       : (게임이름)_X.json 학습 모델 파일이 있는가?

# model0,model1: 상태에 따른 출력 데이터(action별 reward)를 학습한 모델
# last0,last1  : model0, model1의 출력 layer의 index

# gameNo       : 몇 번째 게임인가?
# charSize     : 캐릭터의 크기(가로/세로 길이)
# actions      : 실행할 수 있는 행동의 개수
# exceedValue  : 가능한 범위를 넘어선 경우 적용할, 학습할 screen에서의 값
# charValue    : 캐릭터에 해당하는 부분의 학습할 screen에서의 값

# randomProb_  : 랜덤하게 실행할 확률을 반환하는 함수
# feasible_    : 각 행동이 실행 가능한지 판단하여 True 또는 False를 반환하는 함수
# getNextState_: 다음 상태를 반환하는 함수
def decideAction(option, screen, screenCopy, point, additional, isFile, model0, model1, last0, last1, gameNo,
                 charSize, actions, exceedValue, charValue, randomProb_, feasible_, getNextState_):

    screenSize = len(screen) # screen의 가로/세로 길이
    learningSize = int(math.sqrt(len(screenCopy))) # learningSize (screen의 캐릭터 주변부의 가로/세로 길이)

    around = int((learningSize-charSize)/2) # 캐릭터 주변 8방향으로 몇 칸까지 볼 것인가?
    charRadius = int((charSize-1)/2) # 캐릭터의 중심점에서 주변 8방향으로 몇 칸까지 캐릭터인가?

    # 랜덤하게 실행
    if random.random() < randomProb_(gameNo, option):
        print('랜덤하게 행동 실행')
        while(1):
            action = random.randint(0, actions-1)
            if feasible_(screen, action, additional, point): return action

    # 각 action에 대해 Q(s, a)의 값 구하기
    if isFile:
        # 기본 DQN
        # Q(s_t, a_t)에 대하여 a_t 값을 구해야 하므로 현재 상태인 screenCopy를 이용
        if option % 3 == 0: Qvalues = Qlearning_deep.modelOutput(model0, [screenCopy])

        # Double DQN
        elif option % 3 == 1:
            Qvalues = [-1000000]*actions
            
            for action in range(actions): # 각 행동에 대하여

                # 다음 상태인 newScreen 구하기
                newScreen = getNextState_(screen, action)
                if newScreen == None: # 다음 상태가 없으면(해당 action 실행 결과 index 오류 발생) 고려하지 않기
                    Qvalues[action] == -1000000
                    continue
                
                newPoint = [-1, -1]

                # newScreen에서 캐릭터의 위치 찾기
                for i in range(screenSize):
                    broken = False
                    for j in range(screenSize):
                        if newScreen[i][j] == charValue:
                            newPoint = [i+charRadius, j+charRadius] # 캐릭터의 중심점
                            broken = True
                            break
                    if broken: break

                # newScreen에서 캐릭터 주변만 추출하기
                newScreenCopy = []
                for i in range(newPoint[0]-charRadius-around, newPoint[0]+charRadius+around+1):
                    for j in range(newPoint[1]-charRadius-around, newPoint[1]+charRadius+around+1):
                        # 가능한 범위를 넘어서는 경우 exceedValue를 적용
                        if exceed(newScreen, i, j): newScreenCopy.append(exceedValue)
                        else: newScreenCopy.append(newScreen[i][j])
                
                # model 0에 대한 Q value (action별 reward)
                # Q(s_(t+1), ...)에서 상태가 s_(t+1)이므로 다음 상태인 newScreen을 이용
                Qvalues0 = Qlearning_deep.modelOutput(model0, [newScreenCopy])
                
                # model 1에 대한 Q value (reward가 가장 큰 action만 one-hot으로 1임)
                # Q(s_(t+1), argmax(a)Q(s_(t+1), a))에서 상태가 s_(t+1)이므로 newScreen을 이용
                Qvalues1 = Qlearning_deep.modelOutput(model1, [newScreenCopy])

                # Qvalues1 으로부터 argmax(a)Q(s_(t+1), a)) 구하기
                argmaxA = -1
                for i in range(len(Qvalues1[last0][0])):
                    if Qvalues1[last0][0][i] == max(Qvalues1[last0][0]):
                        argmaxA = i
                        break

                # Qvalues0 으로부터 Q(s_(t+1), argmax(a)Q(s_(t+1), a))의 값 = Qvalue0[argmaxA] 구하기
                Qvalues[action] = Qvalues0[last0][0][argmaxA]

        # Dueling Network
        elif option % 3 == 2: Qvalues = Qlearning_deep.modelOutput(model1, [screenCopy])
        
    else:
        while(1):
            action = random.randint(0, actions-1)
            if feasible_(screen, action, additional, point): return action

    if option % 3 == 1: print('final Qvalues (below is result of action made by them):', Qvalues)

    # 실행할 수 없는 action을 제거
    for i in range(actions):
        if not feasible_(screen, i, additional, point):
            if option % 3 == 1: Qvalues[i] = -1000000
            elif option % 3 == 2: Qvalues[last1][0][i] = -1000000
            else: Qvalues[last0][0][i] = -1000000

    # 보상값이 최대인 행동을 반환
    if option % 3 == 1:
        for i in range(actions):
            if Qvalues[i] == max(Qvalues): return i
    elif option % 3 == 2:
        for i in range(actions):
            if Qvalues[last1][0][i] == max(Qvalues[last1][0]): return i
    else:
        for i in range(actions):
            if Qvalues[last0][0][i] == max(Qvalues[last0][0]): return i

# Neural Network의 상태 입력 및 행동별 보상 출력, 이로 인한 Q value의 변화량을 기록하는 함수

# learningSize : 학습할 영역의 가로/세로 길이
# charSize     : 캐릭터의 크기(가로/세로 길이)

# screen       : 현재 상태
# point        : 현재 상태에서 캐릭터가 있는 지점의 좌표
# lastScreen   : 직전의 상태
# lastDecision : 직전의 상태에서 취한 행동
# lastPoint    : 직전의 상태에서 캐릭터가 있었던 지점의 좌표
# dR           : lastScreen에서 lastDecision을 취했을 때의 즉시 보상

# states       : 상태(Neural Network의 입력값)의 배열
# outputs      : 행동별 보상(Neural Network의 출력값)의 배열
# Qdiffs       : Q value의 변화량의 배열

# exceedValue  : 가능한 범위를 넘어선 경우 적용할, 학습할 screen에서의 값
# isFile       : (게임이름).json 학습 모델 파일이 있는가?

# model0,model1: 상태에 따른 출력 데이터(action별 reward)를 학습한 모델
# last0,last1  : model0, model1의 출력 layer의 index

# func0        : 활성화 함수에 0을 넣었을 때의 값 (예: sigmoid(0)=0.5)
# actions      : 현재 상태에서 취할 수 있는 행동의 개수
# option       : 딥러닝 옵션(0~5)
def logData(learningSize, charSize, screen, point, lastScreen, lastDecision, lastPoint, dR,
            states, outputs, Qdiffs, exceedValue, isFile, model0, model1, last0, last1,
            func0, actions, count, option):

    # 현재 스크린에서 입력으로 넣을 부분만 복사하기
    around = int((learningSize-charSize)/2) # 캐릭터 주변 8방향으로 몇칸까지 학습할 것인가?
    charRadius = int((charSize-1)/2) # 캐릭터의 중심점에서 주변 8방향으로 몇 칸까지 캐릭터인가?

    # ScreenCopy : screen에서 입력으로 넣을 부분
    # screenCopy는 학습할 screen이므로 실제 screen보다 작을수 있음
    screenCopy = arrayCopyFlatten(array=screen,
                                  iStart=point[0]-charRadius-around, iEnd=point[0]+charRadius+around+1,
                                  jStart=point[1]-charRadius-around, jEnd=point[1]+charRadius+around+1,
                                  exceedValue=exceedValue)

    # 카운트가 0이면 lastDecision, lastScreen이 없으므로 학습 불가
    if count > 0:
        # states 입력 추가하기 (lastScreenCopy : lastScreen에서 입력으로 넣을 부분)
        # lastScreenCopy는 학습할 screen이므로 실제 screen보다 작을수 있음
        # (learningSize*learningSize를 flatten)
        lastScreenCopy = arrayCopyFlatten(array=lastScreen,
                                          iStart=lastPoint[0]-charRadius-around, iEnd=lastPoint[0]+charRadius+around+1,
                                          jStart=lastPoint[1]-charRadius-around, jEnd=lastPoint[1]+charRadius+around+1,
                                          exceedValue=exceedValue)

        # 입력값 추가
        states.append(lastScreenCopy)

        # outputs 출력 추가하기
        # 일반 Q learning에서 상태를 검색한다면 deep Q learning에서는 상태를 Neural Network에 넣어서 결과 출력
        output = getQvalues(lastScreenCopy, isFile, model0, model1, last0, last1, func0, actions, option) # 직전 상태에 대한 각 행동의 Q value 구하기
        previousQval = output[lastDecision] # 갱신 이전의 해당 action에 대한 Q value

        # 학습할 output (action의 개수)
        # Q(s, a) <- r + 0.9*max(a')Q(s', a')에서 s' <- screenCopy (현재 screen)임
        output[lastDecision] = sigmoid(dR + 0.9 * maxQvalue(screenCopy, isFile, model0, model1, last0, last1, option))

        # 출력값 추가            
        outputCopy = []
        for j in range(actions): outputCopy.append(output[j])
        outputs.append(outputCopy)

        # Q value의 변화량 추가
        Qdiffs.append(abs(previousQval-output[lastDecision]))

    return screenCopy

# 2차원 배열의 일부를 복사한 후 Flatten시켜 반환하는 함수
# 원래 2차원 배열을 array라 할 때, array[i][j]에서 i, j를 각각 index i, index j라 하자.
# 이때 복사할 범위의 index i는 iStart ~ iEnd-1, index j는 jStart ~ jEnd-1이다.
# array      : 2차원 배열
# iStart     : array의 index i의 시작값
# iEnd       : array의 index i의 종료값
# jStart     : array의 index j의 시작값
# jEnd       : array의 index j의 종료값
# exceedValue: 가능한 범위를 넘어선 경우 적용할, 학습할 screen에서의 값
def arrayCopyFlatten(array, iStart, iEnd, jStart, jEnd, exceedValue):
    result = []
    for i in range(iStart, iEnd):
        for j in range(jStart, jEnd):
            if exceed(array, i, j): result.append(exceedValue) # 가능한 범위를 넘어서는 경우
            else: result.append(array[i][j])
    return result

# 2차원 배열의 전체를 복사하는 함수
def arrayCopy(array):
    assert(len(array) == len(array[0]))

    size = len(array)
    result = [[0] * size for j in range(size)]
    
    for i in range(size):
        for j in range(size):
            result[i][j] = array[i][j]

    return result
