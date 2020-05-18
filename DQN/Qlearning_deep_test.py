import Qlearning_deep
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

if __name__ == '__main__':
    NN = [tf.keras.layers.Flatten(input_shape=(2,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')]
    op = tf.keras.optimizers.Adam(0.001)
    Qdiff = [1, 2, 3, 4, 5, 6, 7]

    # apply each option
    for i in range(6):
        states = [[2, 1], [3, 2], [5, 3], [8, 5], [13, 8], [21, 13], [34, 21]]
        outputs = [[0.2, 0.5], [0.4, 0.6], [0.5, 0.7], [0.7, 0.75], [0.8, 0.77], [0.85, 0.8], [0.9, 0.85]]

        # learning
        print('\n ' + ('#==='*16) + ' <<<< option:' + str(i) + ' LEARNING >>>> ' + ('===#'*16) + '\n')
        Qlearning_deep.deepQlearning(i, NN, op, 'mean_squared_error', states, outputs, Qdiff, 25, 'test' + str(i), [20, 25], [8, 'sigmoid', 'sigmoid'], [6, 'sigmoid', 'sigmoid'], True, True, False)

        # test
        print('\n ' + ('#==='*16) + ' <<<< option:' + str(i) + ' TEST >>>> ' + ('===#'*16) + '\n')
        
        if i % 3 == 0 or i % 3 == 1:
            print('\n << test output (첫번째 Neural Network) >>\n')
            newModel = Qlearning_deep.deepLearningModel('test' + str(i) + '_0', False)
            testOutput = Qlearning_deep.modelOutput(newModel, [[4, 2.5], [6, 3.5], [7, 4.5]])
            print('\n[[4, 2.5], [6, 3.5], [7, 4.5]]에 대한 학습 결과:\n')
            
            for j in range(len(testOutput)):
                print(' << layer No: ' + str(j) + ' >>')
                print(str(testOutput[j]))
                
        if i % 3 == 1 or i % 3 == 2:
            print('\n << test output (두번째(또는 Dueling) Neural Network) >>\n')
            newModel = Qlearning_deep.deepLearningModel('test' + str(i) + '_1', False)
            testOutput = Qlearning_deep.modelOutput(newModel, [[4, 2.5], [6, 3.5], [7, 4.5]])
            print('\n[[4, 2.5], [6, 3.5], [7, 4.5]]에 대한 학습 결과:\n')
            
            for j in range(len(testOutput)):
                print(' << layer No: ' + str(j) + ' >>')
                print(str(testOutput[j]))
