import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

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
    model.add(tf.keras.layers.Dense(43, activation='softmax')) # instead of sigmoid

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
    print(np.shape(train_data))
    print(np.array(train_data))
    print(np.shape(test_data))
    print(np.array(test_data))
    
    print(np.shape(train_input))
    print(np.array(train_input))
    print(np.shape(train_output))
    print(np.array(train_output))
    print(np.shape(test_input))
    print(np.array(test_input))
    print(np.shape(test_output))
    print(np.array(test_output))

    # model training
    model = defineModel()
    model.fit(train_input, train_output, epochs=3, validation_split=0.1)
    model.summary()

    prediction = model.predict(test_input)
    prediction = pd.DataFrame(prediction)
    prediction.to_csv('signTest_prediction.csv')
