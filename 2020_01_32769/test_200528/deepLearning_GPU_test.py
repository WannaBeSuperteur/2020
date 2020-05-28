import deepLearning_GPU
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    
    NN = [tf.keras.layers.Flatten(input_shape=(2,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')]
    op = tf.keras.optimizers.Adam(0.001)

    inputs = [[2, 1], [3, 2], [5, 3], [8, 5], [13, 8], [21, 13], [34, 21]]
    outputs = [[0.2, 0.5], [0.4, 0.6], [0.5, 0.7], [0.7, 0.75], [0.8, 0.77], [0.85, 0.8], [0.9, 0.85]]

    # learning
    print('\n ' + ('#==='*16) + ' <<<< LEARNING >>>> ' + ('===#'*16) + '\n')
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', inputs, outputs, 'test', 500, True, True, deviceName)

    # test
    print('\n ' + ('#==='*16) + ' <<<< TEST >>>> ' + ('===#'*16) + '\n')
        
    print('\n << test output (첫번째 Neural Network) >>\n')
    newModel = deepLearning_GPU.deepLearningModel('test', True)
    testOutput = deepLearning_GPU.modelOutput(newModel, [[4, 2.5], [6, 3.5], [7, 4.5]])
    print('\n[[4, 2.5], [6, 3.5], [7, 4.5]]에 대한 학습 결과:\n')
            
    for i in range(len(testOutput)):
        print(' << layer No: ' + str(i) + ' >>')
        print(str(testOutput[i]))
