import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

# model
class TEXT_MODEL_LSTM0(tf.keras.Model):
    
    def __init__(self, useAtOnce, embed_dim, cnn_filters, dropout_rate=0.25, training=False, name='text_model_LSTM0'):
        
        super(TEXT_MODEL_LSTM0, self).__init__(name=name)

        # constants
        self.use_at_once = useAtOnce

        # flatten layer and regularizer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat    = tf.keras.layers.Flatten()
        L2           = tf.keras.regularizers.l2(0.001)

        # max pooling layers
        self.MP      = tf.keras.layers.MaxPooling1D(pool_size=2)

        # use CNN (CNN2, CNN3, CNN4) + original input
        self.CNN2 = []
        self.CNN3 = []
        self.CNN4 = []
        self.dense_text = []

        for i in range(useAtOnce):
            self.CNN2.append(tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=5, padding='valid', activation='relu', name='CNN2_text_' + str(i)))
            self.CNN3.append(tf.keras.layers.Conv1D(filters=cnn_filters*2, kernel_size=5, padding='valid', activation='relu', name='CNN3_text_' + str(i)))
            self.CNN4.append(tf.keras.layers.Conv1D(filters=cnn_filters*4, kernel_size=5, padding='valid', activation='relu', name='CNN4_text_' + str(i)))

            # layers for inputs_text
            #self.dense_text.append(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_text_' + str(i)))

        # layers for inputs_info
        self.dense_info  = tf.keras.layers.Dense(cnn_filters*4, activation='relu', kernel_regularizer=L2, name='dense_info')

        # layers for final output
        self.merged      = tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=L2, name='merged')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        # suppose that UAO = 10

        ### split inputs
        UAO = self.use_at_once
        i00, i10, i20, i30, i40, i50, i60, i70, i80, i90, inputs_info = tf.split(inputs, [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 3], axis=1)

        ### for text input
        inputs_text = []
        i0s = [i00, i10, i20, i30, i40, i50, i60, i70, i80, i90]
        
        for i in range(UAO):
            i_0 = i0s[i]
            i_0 = tf.reshape(i_0, (-1, 768, 1))

            # CNN layers
            i_1 = self.CNN2[i](i_0)
            i_1 = self.MP(i_1)
            i_2 = self.CNN3[i](i_1)
            i_2 = self.MP(i_2)
            i_3 = self.CNN4[i](i_2)
            i_3 = self.MP(i_3)
            
            i_4 = self.dropout(i_3, training)
            i_4 = self.flat(i_4)
            
            inputs_text.append(i_4)

        ### for info input
        # DNN for inputs_info
        inputs_info = self.dense_info(inputs_info)
        inputs_info = self.dropout(inputs_info, training)
        inputs_info = self.flat(inputs_info)
        
        ### concatenate inputs_text and inputs_info
        concatenated = tf.concat([inputs_text[0], inputs_text[1], inputs_text[2],
                                  inputs_text[3], inputs_text[4], inputs_text[5],
                                  inputs_text[6], inputs_text[7], inputs_text[8],
                                  inputs_text[9], inputs_info], axis=-1)

        # final output
        model_merged = self.merged(concatenated)
        model_output = self.dense_final(model_merged)
        
        return model_output

# model (maxLen is a constant, fill empty index (number of vectors ~ maxLen-1) as 0)
# maxLen : max length with unit 'token'

# [INPUT] -+-> [TEXT] ---> [vec(768)        0] ---> [output        0] -+
#          |               [vec(768)        1] ---> [output        1]  |---> ([CNN 0] + [POOL]) x N ---> [CNN 0 output] -+
#          |               [vec(768)        2] ---> [output        2]  |---> ([CNN 1] + [POOL]) x N ---> [CNN 1 output] -|---> [concatenate] ---> [dense] ---> [FINAL output]
#          |                ...                      ...               |---> ([CNN 2] + [POOL]) x N ---> [CNN 2 output] -+
#          |               [vec(768) maxLen-1] ---> [output maxLen-1] -+                                                 |
#          |                                                                                                             |
#          +-> [INFO] ---------------------------------------------------------------------------------> [Dense] --------+

class TEXT_MODEL_LSTM0_ver01(tf.keras.Model):
    
    def __init__(self, embed_dim, cnn_filters, max_len, dropout_rate=0.25, training=False, name='text_model_LSTM0_ver01'):
        
        super(TEXT_MODEL_LSTM0_ver01, self).__init__(name=name)

        # constants
        self.maxLen      = max_len
        self.N           = 4
        self.cnnFilters  = cnn_filters

        # flatten layer and regularizer
        self.dropout     = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat        = tf.keras.layers.Flatten()
        L2               = tf.keras.regularizers.l2(0.001)

        # CNN (CNN 0, CNN 1 and CNN 2) layers
        self.CNN0 = []
        self.CNN1 = []
        self.CNN2 = []
        
        for i in range(self.N):
            self.CNN0.append(tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=5, padding='valid', activation='relu', name='CNN0_' + str(i)))
            self.CNN1.append(tf.keras.layers.Conv1D(filters=cnn_filters*2, kernel_size=5, padding='valid', activation='relu', name='CNN1_' + str(i)))
            self.CNN2.append(tf.keras.layers.Conv1D(filters=cnn_filters*4, kernel_size=5, padding='valid', activation='relu', name='CNN2_' + str(i)))
            
        self.MP          = tf.keras.layers.MaxPooling1D(pool_size=2, name='POOL') # max pooling layers

        # dense layers for CNN layers
        self.dense_vec   = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=L2, name='dense_vec')
        
        # dense layers for info
        self.dense_info  = tf.keras.layers.Dense(cnn_filters*4, activation='relu', kernel_regularizer=L2, name='dense_info')

        # layer for final output
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):

        maxLength  = self.maxLen
        CNNfilters = self.cnnFilters
        n          = self.N

        # suppose that inputs are FILLED WITH 0 until maxLength

        # split inputs
        TEXT, INFO = tf.split(inputs, [768 * maxLength, 3], axis=1)

        ### for TEXT input
        # vector input -> output
        vectors = tf.split(TEXT, [768 for i in range(maxLength)], axis=1)
        outputs = []

        for i in range(maxLength):
            outputs.append(self.dense_vec(vectors[i]))

        # concatenate
        concat_vecOutputs = tf.concat(outputs, axis=-1)

        # CNN and Pooling layers
        for i in range(n):
            concat_vecOut_reshaped = tf.reshape(concat_vecOutputs, (-1, maxLength, 1))
            #print(i)

            if i == 0:
                concat_vecOut_reshaped_CNN0 = self.CNN0[0](concat_vecOut_reshaped)
                concat_vecOut_reshaped_CNN1 = self.CNN1[0](concat_vecOut_reshaped)
                concat_vecOut_reshaped_CNN2 = self.CNN2[0](concat_vecOut_reshaped)
            else:
                concat_vecOut_reshaped_CNN0 = self.CNN0[i](concat_vecOut_reshaped_CNN0)
                concat_vecOut_reshaped_CNN1 = self.CNN1[i](concat_vecOut_reshaped_CNN1)
                concat_vecOut_reshaped_CNN2 = self.CNN2[i](concat_vecOut_reshaped_CNN2)

            concat_vecOut_reshaped_CNN0 = self.MP(concat_vecOut_reshaped_CNN0)
            concat_vecOut_reshaped_CNN1 = self.MP(concat_vecOut_reshaped_CNN1)
            concat_vecOut_reshaped_CNN2 = self.MP(concat_vecOut_reshaped_CNN2)

        ### for INFO input
        inputs_info = self.dense_info(INFO)
        inputs_info = self.dropout(inputs_info, training)
        inputs_info = tf.reshape(inputs_info, (-1, 1, CNNfilters*4))

        ### final output
        # flatten
        #print('concat_vecOut_reshaped_CNN0:', np.shape(concat_vecOut_reshaped_CNN0))
        #print('concat_vecOut_reshaped_CNN1:', np.shape(concat_vecOut_reshaped_CNN1))
        #print('concat_vecOut_reshaped_CNN2:', np.shape(concat_vecOut_reshaped_CNN2))
        #print('inputs_info:', np.shape(inputs_info))
        
        concat_vecOut_reshaped_CNN0 = self.flat(concat_vecOut_reshaped_CNN0)
        concat_vecOut_reshaped_CNN1 = self.flat(concat_vecOut_reshaped_CNN1)
        concat_vecOut_reshaped_CNN2 = self.flat(concat_vecOut_reshaped_CNN2)
        inputs_info = self.flat(inputs_info)
        
        # merge
        #print('concat_vecOut_reshaped_CNN0:', np.shape(concat_vecOut_reshaped_CNN0))
        #print('concat_vecOut_reshaped_CNN1:', np.shape(concat_vecOut_reshaped_CNN1))
        #print('concat_vecOut_reshaped_CNN2:', np.shape(concat_vecOut_reshaped_CNN2))
        #print('inputs_info:', np.shape(inputs_info))
        
        concatenated = tf.concat([concat_vecOut_reshaped_CNN0,
                                  concat_vecOut_reshaped_CNN1,
                                  concat_vecOut_reshaped_CNN2, inputs_info], axis=-1)

        #print('concatenated:', np.shape(concatenated))

        # final
        model_merged = self.flat(concatenated)
        model_output = self.dense_final(model_merged)
        
        return model_output

class TEXT_MODEL_LSTM1(tf.keras.Model):
    
    def __init__(self, vocab_size, max_length, embed_dim, cnn_filters, dropout_rate=0.25, training=False, name='text_model_LSTM1',
                 use_INFO=True, use_LSTM=True, INFOS=3):
        
        super(TEXT_MODEL_LSTM1, self).__init__(name=name)

        # constants
        self.vocabSize = vocab_size
        self.maxLength = max_length
        self.useINFO   = use_INFO
        self.useLSTM   = use_LSTM
        self.INFOS     = INFOS

        # flatten layer and regularizer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.flat    = tf.keras.layers.Flatten()
        L2           = tf.keras.regularizers.l2(0.001)

        # final dense layer
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=L2, name='dense_final')

        # when using INFO
        if use_INFO == True:
            self.dense_info  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_info')
            self.dense_info1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_info1')
            self.dense_info2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_info2')

        # when using LSTM
        if use_LSTM == True:
            
            # use CNN (CNN2, CNN3, CNN4) + original input
            self.CNN2    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding='valid', activation='relu', name='CNN2_text')
            self.CNN3    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu', name='CNN3_text')
            self.CNN4    = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu', name='CNN4_text')

            # layers for inputs_text
            self.embedding   = tf.keras.layers.Embedding(vocab_size, embed_dim, name='embedding_text')
            self.LSTM        = tf.keras.layers.LSTM(64, name='LSTM_text')
            self.dense_text  = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L2, name='dense_text')

            # layers for final output
            self.merged      = tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=L2, name='merged')

    def call(self, inputs, training):

        # split inputs
        INFOS = self.INFOS
        max_len = self.maxLength
        using_info = self.useINFO
        using_lstm = self.useLSTM

        # when using LSTM
        if using_lstm == True and using_info == True:
            inputs_text, unused, inputs_info = tf.split(inputs, [max_len-1, 1, INFOS], axis=1)

            # CNN layers
            inputs_text = tf.reshape(inputs_text, (-1, max_len-1))

            # Embedding, LSTM, CNN and dense for inputs_text
            inputs_text = self.embedding(inputs_text)
            inputs_text = self.dropout(inputs_text, training)
            
            inputs_text = self.LSTM(inputs_text)
            inputs_text = self.dropout(inputs_text, training)

            # use CNN
            inputs_text = tf.reshape(inputs_text, (-1, 64, 1))
            
            inputs_text_CNN2 = self.CNN2(inputs_text)
            inputs_text_CNN3 = self.CNN3(inputs_text)
            inputs_text_CNN4 = self.CNN4(inputs_text)

            inputs_text_CNN2 = self.flat(inputs_text_CNN2)
            inputs_text_CNN3 = self.flat(inputs_text_CNN3)
            inputs_text_CNN4 = self.flat(inputs_text_CNN4)
            
            inputs_text = tf.concat([inputs_text_CNN2, inputs_text_CNN3, inputs_text_CNN4], axis=-1)
            inputs_text = self.dropout(inputs_text, training)

            # dense after CNN
            inputs_text = self.dense_text(inputs_text)
            inputs_text = self.dropout(inputs_text, training)
            
            # DNN for inputs_info
            inputs_info = self.dense_info(inputs_info)
            inputs_info = self.dropout(inputs_info, training)
            
            # concatenate inputs_text and inputs_info
            concatenated = tf.concat([inputs_text, inputs_info], axis=-1)

            # final output
            model_merged = self.merged(concatenated)
            model_output = self.dense_final(model_merged)

        # do not use LSTM
        elif using_lstm == False and using_info == True:
            unused, inputs_info = tf.split(inputs, [max_len, INFOS], axis=1)

            # DNN for inputs_info
            inputs_info = self.dense_info(inputs_info)
            inputs_info = self.dropout(inputs_info, training)
            inputs_info = self.dense_info1(inputs_info)
            inputs_info = self.dropout(inputs_info, training)
            inputs_info = self.dense_info2(inputs_info)
            inputs_info = self.dropout(inputs_info, training)
            
            model_output = self.dense_final(inputs_info)

        # do not use INFO
        elif using_lstm == True and using_info == False:
            inputs_text, unused = tf.split(inputs, [max_len-1, 1+INFOS], axis=1)

            # CNN layers
            inputs_text = tf.reshape(inputs_text, (-1, max_len-1))

            # Embedding, LSTM, CNN and dense for inputs_text
            inputs_text = self.embedding(inputs_text)
            inputs_text = self.dropout(inputs_text, training)
            
            inputs_text = self.LSTM(inputs_text)
            inputs_text = self.dropout(inputs_text, training)

            # use CNN
            inputs_text = tf.reshape(inputs_text, (-1, 64, 1))
            
            inputs_text_CNN2 = self.CNN2(inputs_text)
            inputs_text_CNN3 = self.CNN3(inputs_text)
            inputs_text_CNN4 = self.CNN4(inputs_text)

            inputs_text_CNN2 = self.flat(inputs_text_CNN2)
            inputs_text_CNN3 = self.flat(inputs_text_CNN3)
            inputs_text_CNN4 = self.flat(inputs_text_CNN4)
            
            inputs_text = tf.concat([inputs_text_CNN2, inputs_text_CNN3, inputs_text_CNN4], axis=-1)
            inputs_text = self.dropout(inputs_text, training)

            # dense after CNN
            inputs_text = self.dense_text(inputs_text)
            inputs_text = self.dropout(inputs_text, training)
            
            # final output
            model_output = self.dense_final(inputs_text)
        
        return model_output
