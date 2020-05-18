import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import Qlearning
import Qlearning_stochastic
import mapReader
from tensorflow.keras.models import Model, model_from_json
from keras import backend as K

### 0. 딥러닝을 위한 Neural Network 생성 함수 ###
# modelInfo: 모델에 대한 정보를 저장한 배열
#            각 원소는 keras.layers.XXX(...) 형식임
# optimi   : Neural Network의 optimizer
#            (예: optimizer.Adam(learning_rate=0.01, beta_1=0.8, bete_2=0.99, amsgrad=False)
#            또는 optimizer의 이름 (예: 'adam')
# isPrint  : 모델 정보 출력 여부
def create_model(modelInfo, optimi, loss, isPrint):
    model = tf.keras.Sequential(modelInfo)
    if isPrint: model.summary()
    model.compile(optimizer=optimi, loss=loss, metrics=['accuracy'])
    return model

### 1. 모델을 학습시키기 ###
# model     : NN 또는 CNN 함수로 생성한 Nerual Network
# states    : Deep Q learning에서 input으로 주어지는 state들을 저장한 배열, 즉 입력 데이터
#             입력데이터가 N개일 때 [[a0, b0, ...], [a1, b1, ...], ..., [a(N-1), b(N-1), ...]] 형태임
# outputs   : Deep Q learning에서의 출력으로 각 Action의 Reward, Reward가 가장 큰 Action 또는 Stream Value가 있는데,
#             이러한 출력들을 저장한 배열, 즉 출력 데이터
#             출력데이터가 N개일 때 [[p0, q0, ...], [p1, q1, ...], ..., [p(N-1), q(N-1), ...]] 형태임
# saveName  : Neural Network 모델 정보 파일(.json, .h5)의 이름
# epoch     : epoch
# deviceName: name of device (ex: 'cpu:0', 'gpu:0')
def learning(model, states, outputs, saveName, epoch, deviceName):

    # 학습 실시
    with tf.device('/' + deviceName):
        model.fit(states, outputs, epochs=epoch)

    # 학습 결과(딥러닝 모델) 저장
    with open(saveName + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(saveName + '.h5')

### 2. Deep Q learning 함수 ###
# option    : 0(기본), 1(DQN), 2(Dueling Network), 3(경험반복), 4(경험반복, DQN) 또는 5(경험반복, Dueling Network)
# NN        : Neural Network에 대한 모델 정보
#             create_model 함수의 modelInfo 인자로 keras.layers.XXX(...) 형식의 원소들로 구성된 배열
# op        : Neural Network의 optimizer
#             create_model 함수의 optimi 인자로 optimizer.XXX(...) 또는 optimizer의 이름(예: 'adam')
# loss      : Neural Network의 loss 계산 방법 (예: 'mean_squared_error')
# states    : Neural Network의 학습 데이터의 states(입력데이터)
#             [[a0, b0, ...], [a1, b1, ...], ..., [a(n-1), b(n-1), ...]] 꼴 (n개의 입력데이터)
# outputs   : Neural Network의 학습 데이터의 outputs(출력데이터)
#             [[p0, q0, ...], [p1, q1, ...], ..., [p(n-1), q(n-1), ...]] 꼴 (n개의 출력데이터)
# Qdiff     : Nerual Network의 해당 학습으로 인해 갱신된 action의 reward 값과 이전 해당 action의 reward 값과의 차이
#             [qd0, qd1, ..., qd(n-1)] 꼴 (n개의 학습데이터에 대해 k번째 학습데이터에 의한 해당 action의 reward 값 차이는 qdk)
# saveName  : Neural Network 모델 정보 파일(.json, .h5)의 이름
# repeat    : 경험 반복을 적용할 때 추가할 학습 데이터의 개수
# epoch     : Neural Network의 학습 epoch의 배열 [epoch0, epoch1]
#             epoch0은 첫번째 NN, epoch1은 두번째 NN의 epoch
# dense0    : Dueling Network의 scalar V(s)에 대한 dense 층의 [뉴런개수, 0번째층 활성화함수, 1번째층 활성화함수] (예: [16, 'sigmoid', 'sigmoid'])
# dense1    : Dueling Network의 vector A(s, a)에 대한 dense 층의 [뉴런개수, 0번째층 활성화함수, 1번째층 활성화함수]
# isPrint   : 관련 정보 출력 여부
# dataPrint : 학습 데이터 출력 여부
# modelPrint: model의 summary 출력 여부
# deviceName: name of device (ex: 'cpu:0', 'gpu:0')
def deepQlearning(option, NN, op, loss, states, outputs, Qdiff, repeat, saveName, epoch, dense0, dense1, isPrint, dataPrint, modelPrint, deviceName):

    dataLen = len(states) # 학습 데이터의 최초 크기

    # 경험 반복 여부
    replay = False # 경험 반복을 할 것인가?
    if option >= 3: replay = True

    # Double DQN, Dueling Network 옵션
    DoubleDQN = False # Double DQN을 사용할 것인가?
    Dueling = False # Dueling Network를 사용할 것인가?
    if option % 3 == 1: DoubleDQN = True
    elif option % 3 == 2: Dueling = True

    if isPrint: print('replay: ' + str(replay) + ' DoubleDQN: ' + str(DoubleDQN) + ' Dueling: ' + str(Dueling))

    ## 2-1. 경험 반복
    if replay:
        # 각 데이터가 뽑힐 확률은 각각의 데이터에 대한 다음 수식으로 한다.
        # 각 데이터의 행동으로 갱신된 Q value와 그 이전의 Q value의 차이의 절댓값 = |qDif| 이면
        # |qDif|^0.6 + 0.01 / Sum(|qDif|^0.6 + 0.01)
        indexLength = 0.0 # 랜덤하게 정해지는 index 값의 범위: [0, indexLength]
        probInfo = [] # [a0=0, a1, ..., a(n-1), indexLength] 꼴의 배열로 index가 ak, a(k+1) 사이에 있으면 k번째 데이터를 뽑음
        for i in range(dataLen):
            probInfo.append(indexLength)
            indexLength += pow(Qdiff[i], 0.6) + 0.01
        probInfo.append(indexLength)

        # 데이터 뽑기 (입력 데이터 states와 출력 데이터 outputs에 k번째 데이터 추가)
        # binary search 알고리즘으로 탐색 시간 줄이기
        for i in range(repeat):
            index = random.random()*indexLength # 데이터를 뽑기 위한 0~indexLength 사이의 랜덤값
            dataNo = int(dataLen/2) # 뽑을 데이터의 번호 k
            dataNoMin = 0 # k 값의 하한선
            dataNoMax = dataLen # k 값의 상한선

            # k번째 데이터 찾기
            while(1):
                # 정확히 일치하면 해당 데이터를 추가하고(뽑고) 종료
                if index >= probInfo[dataNo] and index < probInfo[dataNo+1]:
                    states.append(states[dataNo])
                    outputs.append(outputs[dataNo])
                    Qdiff.append(0) # Qdiff에 dataNo번째 경험(반복되는 경험)에 대한 데이터 추가
                    break
                # index < ak 이면 k의 값을 (0, k-1) 범위 내로 재설정
                elif index < probInfo[dataNo]:
                    dataNoMax = dataNo-1
                    dataNo = int((dataNoMax+dataNoMin)/2)
                # index > a(k+1) 이면 k의 값을 (k+1, n) 범위 내로 재설정
                else:
                    dataNoMin = dataNo+1
                    dataNo = int((dataNoMax+dataNoMin)/2)

    updatedDataLen = len(states) # 학습 데이터의 바뀐 크기

    # 학습 데이터 출력
    if dataPrint:
        print('\n\n< 학습 데이터 >')
        for i in range(updatedDataLen):
            print(str(i) + ': ' + str(states[i]) + ' -> ' + str(outputs[i]))

    ## 2-2. 모델 생성 및 학습
    model = create_model(NN, op, loss, False)
    if not Dueling: # Dueling Network는 따로 학습
        # 모델 정보 출력
        if modelPrint:
            print('\n\n< 첫번째 Neural Network의 구조 >')
            model.summary()
            
        learning(model, [states], [outputs], saveName + '_0', epoch[0], deviceName) # 모델 저장은 이 함수에서 이루어짐

    # outputs 배열에 있는 각 출력 데이터의 크기, 즉 action의 개수
    eachOutputLength = len(outputs[0])

    # 2-3. Double DQN인 경우 첫번째 NN은 action별 reward를, 두번째 NN은 reward가 최대가 되는 action을 학습
    # 딥큐러닝 정확도 높이려면 묻고 더블로 가!!
    if DoubleDQN:
        model_ = create_model(NN, op, loss, False) # 두번째 Neural Network 모델

        # outputs 배열에서 reward가 최대가 되는 action에만 1을 적용
        # 예를 들어 outputs[k] = [0.45, 0.97, 0.26, 0.15] 이면 outputs_[k] = [0, 1, 0, 0]
        outputs_ = []

        for i in range(updatedDataLen):
            temp = [0]*eachOutputLength # outputs_에 삽입할 배열
            maxIndex = 0 # outputs[i]에서 가장 큰 값의 index
            maxValue = -9999 # outputs[i]에서 가장 큰 값
            for j in range(eachOutputLength):
                if outputs[i][j] > maxValue:
                    maxIndex = j
                    maxValue = outputs[i][j]
            temp[maxIndex] = 1 # outputs[i]에서 가장 큰 값의 index에 1을 적용
            outputs_.append(temp)

        # 학습 데이터 출력
        if dataPrint:
            print('\n\n< 두번째 NN의 학습 데이터 >')
            for i in range(updatedDataLen):
                print(str(i) + ': ' + str(states[i]) + ' -> ' + str(outputs_[i]))

        # 모델 정보 출력
        if modelPrint:
            print('\n\n< 두번째 Neural Network의 구조 >')
            model_.summary()

        # 두번째 Neural Network 학습
        learning(model_, [states], [outputs_], saveName + '_1', epoch[1], deviceName) # 모델 저장은 이 함수에서 이루어짐
    
    # 2-4. Dueling Network의 경우 dense 구조를 추가하여 두 stream을 분리하고 마지막에 합산
    # 여기서는 Q(s,a) = V(s)+A(s,a) 수식을 이용하여 Q Table의 값 계산 (V(s)는 output layer의 모든 node에 일괄적용)
    # 실제로는 이렇게 하면 두 stream을 합산했을 때 각각의 특성을 찾을수 없음
    # output layer의 node 개수는 A(s, a)의 두번째 dense층의 node 개수, 즉 action의 개수와 같음 (action별 보상이므로 당연히)
    # 원래 모델 구조에서 마지막 출력레이어는 무시
    if Dueling:
        originalLayerN = len(model.layers) # 원래 model의 레이어 개수

        # V(s)에 대한 dense 층을 기존 모델의 마지막 히든 레이어에 추가
        out = model.layers[originalLayerN-2].output
        out0 = tf.keras.layers.Dense(dense0[0], activation=dense0[1])(out)
        out0 = tf.keras.layers.Dense(1, activation=dense0[2])(out0)

        # A(s,a)에 대한 dense 층을 기존 모델의 마지막 히든 레이어에 추가
        out1 = tf.keras.layers.Dense(dense1[0], activation=dense1[1])(out)
        out1 = tf.keras.layers.Dense(eachOutputLength, activation=dense1[2])(out1)

        # out0과 out1을 합친 레이어 추가 (최종 출력되는 Q 값)
        finalQ = keras.layers.Add()([out0, out1])

        # 최종 모델 생성
        # 입력은 states 값, 출력은 finalQ(s, a) = V(s)+A(s, a)
        model = Model(inputs=model.layers[0].input, outputs=finalQ)

        # 모델 정보 출력
        if modelPrint:
            print('\n\n< Dueling Neural Network의 구조 >')
            model.summary()

        # 학습
        model.compile(optimizer=op, loss=loss, metrics=['accuracy'])
        learning(model, [states], [outputs], saveName + '_1', epoch[1], deviceName) # 모델 저장은 이 함수에서 이루어짐

### 3. 파일로부터 학습된 모델을 불러오는 함수 ###
# saveName: Neural Network 모델 정보 파일(.json, .h5)의 이름
# isPrint : 관련 정보 출력 여부
def deepLearningModel(saveName, isPrint):

    # 기존에 학습한 결과 불러오기
    jsonFile = open(saveName + '.json', 'r')
    loaded_model_json = jsonFile.read()
    jsonFile.close()
    newModel = tf.keras.models.model_from_json(loaded_model_json)
    newModel.load_weights(saveName + '.h5')

    if isPrint: newModel.summary()

    # 모델 컴파일하기
    newModel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])
    return newModel

### 4. 학습된 모델에 값을 입력하여 출력을 구하는 함수 ###
# model    : 학습된 모델
# testInput: 입력 배열 [[a0, b0, ...], [a1, b1, ...], ..., [a(n-1), b(n-1), ...]] 꼴 (n개의 입력데이터)
# simple   : model의 중간에 layer가 갈라지거나 합쳐지는 부분이 없으면 True
def modelOutput(model, testInput):

    # 값을 입력받아 레이어의 출력을 구하도록 함수 설정하기
    input_ = model.input
    output_ = [K.identity(layer.output) for layer in model.layers]

    func = K.function([input_, K.learning_phase()], output_)
    testInput = np.array(testInput)
    result = func([testInput, 1]) # Neural Network에 넣고 결과 받아오기 
    return result # 결과 반환

### 4. 모델을 이용하여 Frozen Lake를 학습시키는 함수 ###
# model       : NN 또는 CNN 함수로 생성한 Nerual Network
# fn          : 학습할 맵의 파일 이름
# isStochastic: deterministic이면 False, Non-deterministic이면 True
# lrStochastic: Non-deterministic을 학습할 때의 학습률
# lrDeepQ     : Deep Q-learning에서의 학습률
# isPrint     : 관련 정보 출력 여부
# deviceName  : name of device (ex: 'cpu:0', 'gpu:0')
def learnFrozenLake(fn, isStochastic, lrStochastic, lrDeepQ, isPrint, deviceName):
    
    ## 4-0. 맵 읽고 Q-table 초기화하기 ##
    info = mapReader.readMap(fn, False) # 맵 읽기
        
    map_ = info[0]
    width = info[1] # 맵의 가로 길이
    height = info[2] # 맵의 세로 길이

    Qtable = mapReader.initQtable(width, height) # Q-table 초기화

    ## 4-1. Q-learning으로 일단 학습 ##
    if isPrint: print('\n\n---< 1. Q-learning으로 학습 >---')
    if isStochastic: data = Qlearning_stochastic.learning(map_, width, height, Qtable, 1, lrStochastic, isPrint)
    else: data = Qlearning.learning(map_, width, height, Qtable, 0, isPrint) # (S, A, R, S) set의 집합

    ## 4-2. 1의 학습 결과 데이터를 사용할 수 있는 데이터로 변환 및 학습 데이터 생성
    # Neural Network로 학습시키기 위한 (state, action별 reward)를 저장하기 위한 배열
    stateInfo = [] # 형식은 [세로좌표][가로좌표][action별 보상]
    for i in range(height):
        temp = []
        for j in range(width):
            temp.append([0, 0, 0, 0])
        stateInfo.append(temp)

    # 학습 데이터 (state, action별 reward) 생성
    states = [] # one-hot 적용
    outputs = []

    # state0, state1에 one-hot을 적용.
    # width=W, height=H이면 state = [w0, w1, ..., w(W-1), h0, h1, ..., h(H-1)]꼴
    # 여기서 현재 위치의 좌표가 (wX, hY)이면 state에서 wX=1, hY=1이고 나머지는 모두 0

    # action에 one-hot을 적용.
    # action = [0, 0, 0, 0]으로 초기화하고 action의 번호가 A이면 action[A]=1로 수정하여 해당 부분만 1로 적용.
    
    for i in range(len(data)):
        # state(t)
        state0 = [0]*(width+height)
        # state(t)의 좌표를 (wX, hY)라고 하면
        state0[data[i][0][1]] = 1 # state(t)의 wX를 1로 적용
        state0[width + data[i][0][0]] = 1 # state(t)의 hY를 1로 적용

        # action(t)
        action0 = [0, 0, 0, 0]
        action0[data[i][1]] = 1

        # reward(t+1)
        reward1 = data[i][2]

        # state(t+1)
        state1 = [0]*(width+height)
        # state(t+1)의 좌표를 (wX, hY)라고 하면
        state1[data[i][3][1]] = 1 # state(t+1)의 wX를 1로 적용
        state1[width + data[i][3][0]] = 1 # state(t+1)의 hY를 1로 적용

        # 행동을 할 때마다 stateInfo 테이블의 action별 reward 갱신
        stateInfo[data[i][0][0]][data[i][0][1]][data[i][1]] = reward1

        if i >= len(data)-800: # 마지막 800개만 학습
            states.append(state0) # 학습 데이터의 state 부분
            temp = []
            for j in range(4):
                rewardVal = stateInfo[data[i][0][0]][data[i][0][1]][j]
                rewardVal = 1 / (1 + math.exp(-rewardVal)) # sigmoid 함수 적용
                temp.append(rewardVal) # 학습 데이터의 action별 reward 부분
            outputs.append(temp)

    # 학습 데이터 일부 출력
    if isPrint:
        print('<학습 데이터 일부>')
        for i in range(50):
            print(str(states[i]) + ': ' + str(np.array(outputs[i])))

    ## 4-3. Deep Q Learning으로 학습 ##
    if isPrint: print('\n\n---< 2. Deep learning으로 학습 >---')

    model = create_model([tf.keras.layers.Flatten(input_shape=(width+height,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(4, activation='sigmoid')],
                         tf.keras.optimizers.Adam(0.001), 'mean_squared_error', isPrint)

    learning(model, [states], [outputs], 'FrozenLake', 4, deviceName)    

    ## 4-4. 테스트 ##
    if isPrint: print('\n\n---<3. 테스트>---')

    # 기존에 학습한 결과 불러오기
    jsonFile = open('FrozenLake.json', 'r')
    loaded_model_json = jsonFile.read()
    jsonFile.close()
    newModel = tf.keras.models.model_from_json(loaded_model_json)
    newModel.load_weights('FrozenLake.h5')
    
    if isPrint: newModel.summary()
    newModel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])

    Qtable = mapReader.initQtable(width, height) # Q-table 초기화
    state = [0, 0] # 상태 초기화

    # Q table의 값들을 model로부터 얻은 보상 값들로 바꾸기
    input_ = newModel.input
    output_ = [layer.output for layer in newModel.layers]
    func = K.function([input_, K.learning_phase()], output_)

    # state [i, j] 들을 입력으로 받음
    testInput = []
    for i in range(height):
        for j in range(width):
            temp = [0]*(width+height) # testInput에 추가할 배열
            temp[j] = 1
            temp[width+i] = 1
            testInput.append(temp)
    testInput = np.array(testInput)

    # state [i, j] 들에 대한 output layer 출력값(action별 보상)의 9.9배를 Q table의 해당 값으로
    result = func([testInput, 1])
    print('\n\n< deep-learned Q-table >')
    for i in range(height):
        for j in range(width):
            for k in range(4): Qtable[i][j][k] = result[3][i*width+j][k] * 9.9
            if isPrint: print('Qtable index ' + str(i) + ',' + str(j) + ' -> ' + str(result[3][i*width+j]))

    # 실행 및 맵 출력
    if isStochastic: Qlearning_stochastic.execute(map_, width, height, Qtable, True)
    else: Qlearning.execute(map_, width, height, Qtable, True)

if __name__ == '__main__':
    learnFrozenLake('map.txt', True, 0.8, 0.001, True, input('device name (for example, cpu:0 or gpu:0)'))
