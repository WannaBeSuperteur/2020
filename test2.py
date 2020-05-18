import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras import backend as K

# create and return model for deep learning
def create_model(modelInfo, optimi, isPrint):
    model = tf.keras.Sequential(modelInfo)
    if isPrint: model.summary()
    model.compile(optimizer=optimi, loss='mean_squared_error', metrics=['accuracy'])

    model.json = model.to_json()
    
    return model

def learning(model, states, outputs, saveName, epoch):

    # learning
    print(np.shape(states))
    model.fit(states, outputs, epochs=epoch)

    # save result of learning (model)
    with open(saveName + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(saveName + '.h5')

x = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
y = [[0.25, 0.21], [0.23, 0.64], [0.45, 0.17], [0.46, 0.58], [0.65, 0.15], [0.6, 0.53]]

# create model
model = create_model([keras.layers.Dense(4, activation='tanh', input_dim=2),
        keras.layers.Dense(6, activation='sigmoid'),
        keras.layers.Dense(6, activation='tanh'),
        keras.layers.Dense(6, activation='sigmoid'),
        keras.layers.Dense(2, activation='softmax')],
                         tf.keras.optimizers.Adam(0.015), True)

# repeat learning process 8 times
for i in range(8):
    learning(model, [x], [y], 'test', 10)

    # read model
    jsonFile = open('test.json', 'r')
    loaded_model_json = jsonFile.read()
    jsonFile.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights('test.h5')
    model = loaded_model
    
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.015), loss='mean_squared_error', metrics=['accuracy'])

# test
input_ = model.input
output_ = [layer.output for layer in model.layers]
func = K.function([input_, K.learning_phase()], output_)

testInput = np.array([[3, 0], [3, 1], [4, 0], [4, 1], [5, 0], [5, 1]]) # test input
result = func([testInput, 1.]) # print the output of test input

print('test input:')
print(str(testInput))
for i in range(len(result)):
    print('output of layer' + str(i) + ':')
    print(str(result[i])) 
