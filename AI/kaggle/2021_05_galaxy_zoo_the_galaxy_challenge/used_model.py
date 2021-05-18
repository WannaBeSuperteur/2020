import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

class MODEL(tf.keras.Model):
    
    def __init__(self, SIZE=36, OUTPUTS=37, dropout_rate=0.25, training=False, name='model'):
        
        super(MODEL, self).__init__(name=name)

        # L2 regularization
        L2 = tf.keras.regularizers.l2(0.001)

        # constants
        self.SIZE = SIZE
        self.OUTPUTS = OUTPUTS

        # input part
        self.flat = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape((SIZE, SIZE, 1), input_shape=(SIZE*SIZE,))

        # each channel (R, G and B)
        self.dropout1R = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout1R')
        self.dropout1G = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout1G')
        self.dropout1B = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout1B')
        self.dropout2R = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout2R')
        self.dropout2G = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout2G')
        self.dropout2B = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout2B')

        self.pool1R = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1R')
        self.pool2R = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2R')
        self.pool1G = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1G')
        self.pool2G = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2G')
        self.pool1B = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1B')
        self.pool2B = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2B')

        self.cnn1R = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1R')
        self.cnn2R = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn2R')
        self.cnn3R = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn3R')
        self.cnn4R = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn4R')
        self.cnn5R = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5R')

        self.cnn1G = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1G')
        self.cnn2G = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn2G')
        self.cnn3G = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn3G')
        self.cnn4G = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn4G')
        self.cnn5G = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5G')

        self.cnn1B = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1B')
        self.cnn2B = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn2B')
        self.cnn3B = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn3B')
        self.cnn4B = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn4B')
        self.cnn5B = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5B')

        # after concatenation
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout3')
        self.dense1 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=L2, name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=L2, name='dense2')
        self.dense_out = tf.keras.layers.Dense(units=OUTPUTS, activation='sigmoid', kernel_regularizer=L2, name='dense_out')

    def call(self, inputs, training):

        # split input
        input_R, input_G, input_B = tf.split(inputs, [self.SIZE*self.SIZE, self.SIZE*self.SIZE, self.SIZE*self.SIZE], 1)
            
        # Reshape into (SIZE, SIZE)
        input_R = self.reshape(input_R)
        input_G = self.reshape(input_G)
        input_B = self.reshape(input_B)

        # Convolutional-Pooling layers for R
        input_R = self.cnn1R(input_R)
        input_R = self.dropout1R(input_R, training)
        
        input_R = self.cnn2R(input_R)
        input_R = self.pool1R(input_R)
        input_R = self.cnn3R(input_R)
        input_R = self.dropout2R(input_R, training)
        
        input_R = self.cnn4R(input_R)
        input_R = self.pool2R(input_R)
        input_R = self.cnn5R(input_R)

        # flatten for R
        input_R = self.flat(input_R)

        # Convolutional-Pooling layers for G
        input_G = self.cnn1G(input_G)
        input_G = self.dropout1G(input_G, training)
        
        input_G = self.cnn2G(input_G)
        input_G = self.pool1G(input_G)
        input_G = self.cnn3G(input_G)
        input_G = self.dropout2G(input_G, training)
        
        input_G = self.cnn4G(input_G)
        input_G = self.pool2G(input_G)
        input_G = self.cnn5G(input_G)

        # flatten for G
        input_G = self.flat(input_G)

        # Convolutional-Pooling layers for B
        input_B = self.cnn1B(input_B)
        input_B = self.dropout1B(input_B, training)
        
        input_B = self.cnn2B(input_B)
        input_B = self.pool1B(input_B)
        input_B = self.cnn3B(input_B)
        input_B = self.dropout2B(input_B, training)
        
        input_B = self.cnn4B(input_B)
        input_B = self.pool2B(input_B)
        input_B = self.cnn5B(input_B)

        # flatten for R
        input_B = self.flat(input_B)

        # concatenate flatten inputs
        input_concat = tf.concat([input_R, input_G, input_B], axis=-1)

        # dense and output layers (fully-connected part)
        input_concat = self.dense1(input_concat)
        input_concat = self.dropout3(input_concat, training)
        input_concat = self.dense2(input_concat)
        model_output = self.dense_out(input_concat)
        
        return model_output
