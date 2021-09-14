# ref: https://huggingface.co/transformers/model_doc/roberta.html

import random
import numpy as np
import pandas as pd
import datetime
import re

# if error, refer to https://www.gitmemory.com/issue/huggingface/transformers/3442/756067404
#           or install tensorflow (or tensorflow-gpu) 2.4.1
from transformers import RobertaTokenizer, TFRobertaModel

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# option: [op0, op1, ..., opN] for each column
# each option is:
# -1 : except for this column
#  0 : numeric
#  1 : using roberta model
#      the shape of result is ((tokens, 768)) and using the first element for each token

def extract(fn, option, title, roberta_tokenizer, roberta_model, roberta_max_tokens):

    array = np.array(pd.read_csv(fn))
    print(array)
    
    rows = len(array)
    cols = len(array[0])
    print(rows, cols)
    
    resultArray = []

    # for each row
    newTitle = []
    for i in range(rows):
        if i % 1 == 0: print(i)
        
        thisRow = []

        # for each column
        for j in range(cols):
            max_tokens = roberta_max_tokens[j]

            # -1 : except for this column
            if option[j] == -1:
                pass

            #  0 : numeric
            elif option[j] == 0:
                thisRow.append(float(array[i][j]))

                if i == 0: newTitle.append(str(j) + '_val')

            #  1 : using roBERTa model
            elif option[j] == 1:

                # get roBERTa last hidden state
                text = array[i][j]
                roberta_lhs = getFirstRobertaLHS(text, roberta_tokenizer, roberta_model)

                # append encoded text
                for k in range(len(roberta_lhs)):
                    thisRow.append(roberta_lhs[k][0])

                # fill empty places before encoded text
                for k in range(max_tokens - len(roberta_lhs)):
                    thisRow.append(0, 0)

                # title
                if i == 0:
                    for k in range(max_tokens):
                        thisRow.append(str(j) + '_roberta_' + str(k))

        resultArray.append(thisRow)

    return (resultArray, newTitle, onehot)

# get first element of roBERTa last hidden state
def getFirstRobertaLHS(text, roberta_tokenizer, roberta_model):
    roberta_input  = roberta_tokenizer(text, return_tensors='tf')
    roberta_output = roberta_model(roberta_input)
    roberta_lhs    = roberta_output.last_hidden_state # shape: (1, N, 768)
    roberta_lhs    = np.array(roberta_lhs)[0] # shape: (N, 768)

    return roberta_lhs

# encode text using roBERTa tokenizer and roBERTa model
def encodeText(text, roberta_tokenizer, roberta_model):
    regex = re.compile('[^a-zA-Z0-9.,?! ]')

    try:
        # use first 25 characters
        text = regex.sub('', text)[:25]
        roberta_lhs = getFirstRobertaLHS(text, roberta_tokenizer, roberta_model)
    except:
        roberta_lhs = np.zeros((1, 768))
        
    encoded = []
    for i in range(len(roberta_lhs)):
        encoded.append(roberta_lhs[i][0])
    return encoded

# find the max count of the number of roBERTa tokens for each column (project_title, ..., project_resource_summary)
def get_roberta_token_count(roberta_tokenizer, roberta_model):

    # read saved max tokens count
    try:
        roberta_max_tokens = pd.read_csv('roberta_max_tokens.csv', index_col=0)
        return np.array(roberta_max_tokens)

    # new max tokens count
    except:
        titles = ['project_title', 'project_essay_1', 'project_essay_2',
                  'project_essay_3', 'project_essay_4', 'project_resource_summary']
        
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        roberta_max_tokens = [0, 0, 0, 0, 0, 0]

        # for training data
        print('find roberta token count for training data ...')
        for i in range(len(train_data)):
            if i % 1 == 0: print(i)
            
            for j in range(6):
                roberta_max_tokens[j] = max(roberta_max_tokens[j],
                                            len(encodeText(train_data.iloc[i][titles[j]],
                                                           roberta_tokenizer, roberta_model)))

        # for test data
        print('find roberta token count for test data ...')
        for i in range(len(test_data)):
            if i % 1 == 0: print(i)
            
            for j in range(6):
                roberta_max_tokens[j] = max(roberta_max_tokens[j],
                                            len(encodeText(test_data.iloc[i][titles[j]],
                                                           roberta_tokenizer, roberta_model)))

        roberta_max_tokens_ = pd.DataFrame(np.array(roberta_max_tokens))
        roberta_max_tokens_.to_csv('roberta_max_tokens.csv')

        return np.array(roberta_max_tokens)

if __name__ == '__main__':

    train_option = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 0]
    test_option = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1]

    # GET ROBERTA TOKEN COUNT
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    
    roberta_max_tokens = get_roberta_token_count(roberta_tokenizer, roberta_model)

    # TRAINING
    
    # extract data
    train_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    (train_extracted, train_newTitle, onehot) = extract('train.csv', train_option, train_title, roberta_tokenizer, roberta_model, roberta_max_tokens)

    train_newTitle_input = train_newTitle[:len(train_newTitle)-1]
    train_newTitle_output = [train_newTitle[len(train_newTitle)-1]]

    # divide into input and output
    train_input = [] # [train_newTitle_input]
    train_output = [] # [train_newTitle_output]
    
    for i in range(len(train_extracted)):

        # move target value to the first column
        train_lenm1 = len(train_extracted[i])-1

        train_input.append(train_extracted[i][:train_lenm1])
        train_output.append([train_extracted[i][train_lenm1]])
        
    train_input = pd.DataFrame(np.array(train_input))
    train_output = pd.DataFrame(np.array(train_output))
    train_input.to_csv('train_input_text.csv')
    train_output.to_csv('train_output_text.csv')

    # TEST
    
    # extract data
    test_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects']

    (test_extracted, test_newTitle, _) = extract('test.csv', test_option, test_title, roberta_tokenizer, roberta_model, roberta_max_tokens)

    test_toWrite = [] # [test_newTitle]
    
    for i in range(len(test_extracted)):
        test_toWrite.append(test_extracted[i])
        
    test_toWrite = pd.DataFrame(np.array(test_toWrite))
    test_toWrite.to_csv('test_input_text.csv')
