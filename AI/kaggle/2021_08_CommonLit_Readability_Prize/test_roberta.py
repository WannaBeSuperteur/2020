# ref: https://wikidocs.net/45101
#      https://huggingface.co/transformers/model_doc/roberta.html

import math
import numpy as np
import pandas as pd
import re

from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

# to execute
if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = TFRobertaModel.from_pretrained('roberta-base')

    input0 = tokenizer("Hello, my dog is cute", return_tensors="tf")
    input1 = tokenizer("Hello, my cat is cute", return_tensors="tf")
    input2 = tokenizer("Hello, my dragon is cute", return_tensors="tf")
    input3 = tokenizer("Hello, my hamster is cute", return_tensors="tf")
    input4 = tokenizer("I will meet my friend at 7 pm.", return_tensors="tf")
    
    output0 = model(input0)
    output1 = model(input1)
    output2 = model(input2)
    output3 = model(input3)
    output4 = model(input4)

    last_hidden_states0 = output0.last_hidden_state
    last_hidden_states1 = output1.last_hidden_state
    last_hidden_states2 = output2.last_hidden_state
    last_hidden_states3 = output3.last_hidden_state
    last_hidden_states4 = output4.last_hidden_state

    lhs_array0 = np.round_(np.array(last_hidden_states0), 4)
    lhs_array1 = np.round_(np.array(last_hidden_states1), 4)
    lhs_array2 = np.round_(np.array(last_hidden_states2), 4)
    lhs_array3 = np.round_(np.array(last_hidden_states3), 4)
    lhs_array4 = np.round_(np.array(last_hidden_states4), 4)

    print('\n\nnumpy array of last hidden states:')
    print('----')
    print('shape: ' + str(np.shape(lhs_array0)))
    print(lhs_array0)
    print('----')
    print('shape: ' + str(np.shape(lhs_array1)))
    print(lhs_array1)
    print('----')
    print('shape: ' + str(np.shape(lhs_array2)))
    print(lhs_array2)
    print('----')
    print('shape: ' + str(np.shape(lhs_array3)))
    print(lhs_array3)
    print('----')
    print('shape: ' + str(np.shape(lhs_array4)))
    print(lhs_array4)

    concatenated = np.concatenate((lhs_array0[0],
                                   lhs_array1[0],
                                   lhs_array2[0],
                                   lhs_array3[0],
                                   lhs_array4[0]),
                         axis=0)

    print('\n\nfinal array:')
    print('shape: ' + str(np.shape(concatenated)))
    print('----')
    print(concatenated)

    concatenated = pd.DataFrame(concatenated)
    concatenated.to_csv('finalArray.csv')
