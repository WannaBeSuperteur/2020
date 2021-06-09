import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json

# print model configuration
def printModelConfig(model, optimi, loss):
    model.summary()
    print('optimizer:')
    print(optimi)
    print('\nloss function:')
    print(loss)
    print('\n')

### 0. 딥러닝을 위한 Neural Network 생성 함수 ###
# modelInfo: 모델에 대한 정보를 저장한 배열
#            각 원소는 keras.layers.XXX(...) 형식임
# optimi   : Neural Network의 optimizer
#            (예: optimizer.Adam(learning_rate=0.01, beta_1=0.8, bete_2=0.99, amsgrad=False)
#            또는 optimizer의 이름 (예: 'adam')
# isPrint  : 모델 정보 출력 여부
def create_model(modelInfo, optimi, loss, isPrint):
    model = tf.keras.Sequential(modelInfo)
    if isPrint: printModelConfig(model, optimi, loss)
    model.compile(optimizer=optimi, loss=loss, metrics=['accuracy'])
    return model

### 1. 모델을 학습시키기 ###
# model     : NN 또는 CNN 함수로 생성한 Nerual Network
# inputs    : Deep Learning에서 input으로 주어지는 state들을 저장한 배열, 즉 입력 데이터
#             입력데이터가 N개일 때 [[a0, b0, ...], [a1, b1, ...], ..., [a(N-1), b(N-1), ...]] 형태임
# outputs   : Deep Learning에서의 출력으로 각 Action의 Reward, Reward가 가장 큰 Action 또는 Stream Value가 있는데,
#             이러한 출력들을 저장한 배열, 즉 출력 데이터
#             출력데이터가 N개일 때 [[p0, q0, ...], [p1, q1, ...], ..., [p(N-1), q(N-1), ...]] 형태임
# saveName  : Neural Network 모델 정보 파일(.json, .h5)의 이름
# epoch     : epoch
# deviceName: name of device (ex: 'cpu:0', 'gpu:0')
def learning(model, inputs, outputs, saveName, epoch, deviceName):

    if len(inputs) > 10:
        print(np.shape(inputs))
        print(np.array(inputs[:5]))
        print('...')
        print(np.array(inputs[len(inputs)-5:]))
    else:
        print(np.shape(inputs))
        print(np.array(inputs))
    
    if len(outputs) > 10:
        print(np.shape(outputs))
        print(np.array(outputs[:5]))
        print('...')
        print(np.array(outputs[len(outputs)-5:]))
    else:
        print(np.shape(outputs))
        print(np.array(outputs))

    # 학습 실시
    with tf.device('/' + deviceName):
        model.fit(inputs, outputs, epochs=epoch)

    # 학습 결과(딥러닝 모델) 저장
    with open(saveName + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(saveName + '.h5')

### 2. Deep Learning 함수 ###
# NN        : Neural Network에 대한 모델 정보
#             create_model 함수의 modelInfo 인자로 keras.layers.XXX(...) 형식의 원소들로 구성된 배열
# op        : Neural Network의 optimizer
#             create_model 함수의 optimi 인자로 optimizer.XXX(...) 또는 optimizer의 이름(예: 'adam')
# loss      : Neural Network의 loss 계산 방법 (예: 'mean_squared_error')
# inputs    : Neural Network의 학습 데이터의 inputs(입력데이터)
#             [[a0, b0, ...], [a1, b1, ...], ..., [a(n-1), b(n-1), ...]] 꼴 (n개의 입력데이터)
# outputs   : Neural Network의 학습 데이터의 outputs(출력데이터)
#             [[p0, q0, ...], [p1, q1, ...], ..., [p(n-1), q(n-1), ...]] 꼴 (n개의 출력데이터)
# saveName  : Neural Network 모델 정보 파일(.json, .h5)의 이름
# epoch     : Neural Network의 학습 epoch
# dataPrint : 학습 데이터 출력 여부
# modelPrint: model의 summary 출력 여부
# deviceName: name of device (ex: 'cpu:0', 'gpu:0')
def deepLearning(NN, op, loss, inputs, outputs, saveName, epoch, dataPrint, modelPrint, deviceName):

    dataLen = len(inputs) # 학습 데이터의 최초 크기

    # 학습 데이터 출력
    if dataPrint:
        print('\n\n< 학습 데이터 >')
        for i in range(dataLen):
            print(str(i) + ': ' + str(inputs[i]) + ' -> ' + str(outputs[i]))

    ## 2-2. 모델 생성 및 학습
    model = create_model(NN, op, loss, False)

    # 모델 정보 출력
    if modelPrint:
        print('\n\n< Neural Network의 구조 >')
        printModelConfig(model, op, loss)
            
    learning(model, inputs, outputs, saveName, epoch, deviceName) # 모델 저장은 이 함수에서 이루어짐

### 3. 파일로부터 학습된 모델을 불러오는 함수 ###
# op      : optimizer
# loss    : loss
# saveName: Neural Network 모델 정보 파일(.json, .h5)의 이름
# isPrint : 관련 정보 출력 여부
def deepLearningModel(saveName, op, loss, isPrint):

    # 기존에 학습한 결과 불러오기
    jsonFile = open(saveName + '.json', 'r')
    loaded_model_json = jsonFile.read()
    jsonFile.close()
    newModel = tf.keras.models.model_from_json(loaded_model_json)
    newModel.load_weights(saveName + '.h5')

    if isPrint: printModelConfig(newModel, op, loss)

    # 모델 컴파일하기
    newModel.compile(optimizer=op, loss=loss, metrics=['accuracy'])
    return newModel

### 4. 학습된 모델에 값을 입력하여 출력을 구하는 함수 ###
# model        : 학습된 모델
# testInput    : 입력 배열 [[a0, b0, ...], [a1, b1, ...], ..., [a(n-1), b(n-1), ...]] 꼴 (n개의 입력데이터)
def modelOutput(model, testInput):

    testInput = np.array(testInput)
    result = model.predict(testInput)
    
    return [result] # 결과 반환
