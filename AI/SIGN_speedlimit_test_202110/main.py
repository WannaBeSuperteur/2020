import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error

# define model (the number of input units not needed)
def defineModel(learningRate=0.001):

    # L2 regularization
    L2 = tf.keras.regularizers.l2(0.001)

    # define the model
    model = Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape((64, 64, 1), input_shape=(64*64,)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(40, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(40, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(40, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='softmax')) # instead of sigmoid

    adamOpt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # instead of mse

    return model

# main
if __name__ == '__main__':

    # meta info
    train_file = 'train.csv'
    test_file = 'test.csv'

    # load data
    train_data = np.array(pd.read_csv(train_file, index_col=0))
    test_data = np.array(pd.read_csv(test_file, index_col=0))
    train_rows = len(train_data)
    test_rows = len(test_data)

    print(np.shape(train_data))
    print(np.array(train_data))

    train_input = np.round_(train_data[:, 1:] / 255.0, 3)
    train_output = train_data[:, :1]
    test_input = np.round_(test_data[:, 1:] / 255.0, 3)
    test_output = test_data[:, :1]

    # convert outputs into one-hot vector
    train_n = np.unique(train_output, axis=0)
    train_n = train_n.shape[0]
    train_output = np.eye(train_n)[train_output]
    
    train_output = np.reshape(train_output, (train_output.shape[0], train_output.shape[2]))

    test_n = np.unique(test_output, axis=0)
    test_n = test_n.shape[0]
    test_output = np.eye(test_n)[test_output]

    test_output = np.reshape(test_output, (test_output.shape[0], test_output.shape[2]))

    # print
    print('train_data:')
    print(np.shape(train_data))
    print(np.array(train_data))

    print('\ntest_data:')
    print(np.shape(test_data))
    print(np.array(test_data))

    print('\ntrain_input:')
    print(np.shape(train_input))
    print(np.array(train_input))

    print('\ntrain_output:')
    print(np.shape(train_output))
    print(np.array(train_output))

    print('\ntest_input:')
    print(np.shape(test_input))
    print(np.array(test_input))

    print('\ntest_output:')
    print(np.shape(test_output))
    print(np.array(test_output))

    pd.DataFrame(test_output).to_csv('signSpeedLimitTest_groundTruth.csv')

    # model training
    try:
        model = tf.keras.models.load_model('signSpeedLimitTest_model')
    except:
        model = defineModel()
        model.fit(train_input, train_output, epochs=10, validation_split=0.1)
        model.save('signSpeedLimitTest_model')
    model.summary()

    prediction = model.predict(test_input)
    prediction = pd.DataFrame(prediction)
    prediction.to_csv('signSpeedLimitTest_prediction.csv')

    # compare test groundTruth and test prediction
    test_output_to_compare = np.array(test_output)
    prediction_to_compare = np.array(prediction)

    # compute mean-square error and accuracy
    MSE = 0.0
    accurateRows = 0

    for i in range(test_rows):
        ground_truth = test_output_to_compare[i]
        pred = prediction_to_compare[i]
        
        MSE += mean_squared_error(ground_truth, pred)
        if np.argmax(ground_truth) == np.argmax(pred): accurateRows += 1

    MSE /= test_rows
    accurateRows /= test_rows
    
    print('MSE      = ' + str(MSE))
    print('accuracy = ' + str(accurateRows))
