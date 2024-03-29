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
        self.dropout3R = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout3R')
        self.dropout3G = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout3G')
        self.dropout3B = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout3B')

        self.pool1R = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1R')
        self.pool2R = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2R')
        self.pool1G = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1G')
        self.pool2G = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2G')
        self.pool1B = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling1B')
        self.pool2B = tf.keras.layers.MaxPooling2D(pool_size=2, name='pooling2B')

        self.cnn1R = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1R')
        self.cnn2R_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu', name='cnn2R_1')
        self.cnn2R_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='cnn2R_2')
        self.cnn2R_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', name='cnn2R_3')
        self.cnn3R_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu', name='cnn3R_1')
        self.cnn3R_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu', name='cnn3R_2')
        self.cnn3R_3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), padding='same', activation='relu', name='cnn3R_3')
        self.cnn4R_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='cnn4R_1')
        self.cnn4R_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='cnn4R_2')
        self.cnn4R_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='cnn4R_3')
        self.cnn5R = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5R')

        self.cnn1G = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1G')
        self.cnn2G_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu', name='cnn2G_1')
        self.cnn2G_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='cnn2G_2')
        self.cnn2G_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', name='cnn2G_3')
        self.cnn3G_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu', name='cnn3G_1')
        self.cnn3G_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu', name='cnn3G_2')
        self.cnn3G_3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), padding='same', activation='relu', name='cnn3G_3')
        self.cnn4G_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='cnn4G_1')
        self.cnn4G_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='cnn4G_2')
        self.cnn4G_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='cnn4G_3')
        self.cnn5G = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5G')

        self.cnn1B = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn1B')
        self.cnn2B_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu', name='cnn2B_1')
        self.cnn2B_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='cnn2B_2')
        self.cnn2B_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', name='cnn2B_3')
        self.cnn3B_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu', name='cnn3B_1')
        self.cnn3B_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu', name='cnn3B_2')
        self.cnn3B_3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), padding='same', activation='relu', name='cnn3B_3')
        self.cnn4B_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='cnn4B_1')
        self.cnn4B_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='cnn4B_2')
        self.cnn4B_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='cnn4B_3')
        self.cnn5B = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', name='cnn5B')

        # after concatenation
        self.dropout4 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout4')
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
        
        input_R1 = self.cnn2R_1(input_R)
        input_R2 = self.cnn2R_2(input_R)
        input_R3 = self.cnn2R_3(input_R)
        input_R = tf.concat([input_R1, input_R2, input_R3], axis=-1)
        
        input_R = self.pool1R(input_R)
        
        input_R1 = self.cnn3R_1(input_R)
        input_R2 = self.cnn3R_2(input_R)
        input_R3 = self.cnn3R_3(input_R)
        input_R = tf.concat([input_R1, input_R2, input_R3], axis=-1)
        
        input_R = self.dropout2R(input_R, training)
        
        input_R1 = self.cnn4R_1(input_R)
        input_R2 = self.cnn4R_2(input_R)
        input_R3 = self.cnn4R_3(input_R)
        input_R = tf.concat([input_R1, input_R2, input_R3], axis=-1)
        
        input_R = self.pool2R(input_R)
        input_R = self.cnn5R(input_R)
        input_R = self.dropout3R(input_R, training)

        # flatten for R
        input_R = self.flat(input_R)

        # Convolutional-Pooling layers for G
        input_G = self.cnn1G(input_G)
        input_G = self.dropout1G(input_G, training)
        
        input_G1 = self.cnn2G_1(input_G)
        input_G2 = self.cnn2G_2(input_G)
        input_G3 = self.cnn2G_3(input_G)
        input_G = tf.concat([input_G1, input_G2, input_G3], axis=-1)
        
        input_G = self.pool1G(input_G)
        
        input_G1 = self.cnn3G_1(input_G)
        input_G2 = self.cnn3G_2(input_G)
        input_G3 = self.cnn3G_3(input_G)
        input_G = tf.concat([input_G1, input_G2, input_G3], axis=-1)
        
        input_G = self.dropout2G(input_G, training)
        
        input_G1 = self.cnn4G_1(input_G)
        input_G2 = self.cnn4G_2(input_G)
        input_G3 = self.cnn4G_3(input_G)
        input_G = tf.concat([input_G1, input_G2, input_G3], axis=-1)
        
        input_G = self.pool2G(input_G)
        input_G = self.cnn5G(input_G)
        input_G = self.dropout3G(input_G, training)

        # flatten for G
        input_G = self.flat(input_G)

        # Convolutional-Pooling layers for B
        input_B = self.cnn1B(input_B)
        input_B = self.dropout1B(input_B, training)
        
        input_B1 = self.cnn2B_1(input_B)
        input_B2 = self.cnn2B_2(input_B)
        input_B3 = self.cnn2B_3(input_B)
        input_B = tf.concat([input_B1, input_B2, input_B3], axis=-1)
        
        input_B = self.pool1B(input_B)
        
        input_B1 = self.cnn3B_1(input_B)
        input_B2 = self.cnn3B_2(input_B)
        input_B3 = self.cnn3B_3(input_B)
        input_B = tf.concat([input_B1, input_B2, input_B3], axis=-1)
        
        input_B = self.dropout2B(input_B, training)
        
        input_B1 = self.cnn4B_1(input_B)
        input_B2 = self.cnn4B_2(input_B)
        input_B3 = self.cnn4B_3(input_B)
        input_B = tf.concat([input_B1, input_B2, input_B3], axis=-1)
        
        input_B = self.pool2B(input_B)
        input_B = self.cnn5B(input_B)
        input_B = self.dropout3B(input_B, training)

        # flatten for R
        input_B = self.flat(input_B)

        # concatenate flatten inputs
        input_concat = tf.concat([input_R, input_G, input_B], axis=-1)

        # dense and output layers (fully-connected part)
        input_concat = self.dense1(input_concat)
        input_concat = self.dropout4(input_concat, training)
        input_concat = self.dense2(input_concat)
        model_output = self.dense_out(input_concat)
        
        return model_output
