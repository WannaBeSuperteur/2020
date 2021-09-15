# ref: https://analyticsindiamag.com/python-guide-to-huggingface-distilbert-smaller-faster-cheaper-distilled-bert/

import random
import numpy as np
import pandas as pd
import datetime
import re

from transformers import DistilBertTokenizer, DistilBertModel
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# option: [op0, op1, ..., opN] for each column
# each option is:
# -1 : except for this column
#  0 : numeric
#  1 : using DistilBert model
#      the shape of result is ((tokens)) and use all these tokens

def extract(fn, option, title, DistilBert_tokenizer, DistilBert_model, DistilBert_max_tokens):

    array = np.array(pd.read_csv(fn))
    print(array)
    
    rows = len(array)
    cols = len(array[0])
    print(rows, cols)
    
    resultArray = []

    # for each row
    newTitle = []
    for i in range(rows):
        if i % 250 == 0: print(i)
        
        thisRow = []

        # for each column
        for j in range(cols):

            # -1 : except for this column
            if option[j] == -1:
                pass

            #  0 : numeric
            elif option[j] == 0:
                thisRow.append(float(array[i][j]))

                if i == 0: newTitle.append(str(j) + '_val')

            #  1 : using DistilBert model
            elif option[j] == 1:

                currentThisRowLength = len(thisRow)

                # cols: {..., 8:project_title, 9:project_essay_1, ...,
                #        12:project_essay_4, 13:project_resource_summary, ...}
                j_textCol = j - 8
                j_encodedCols = DistilBert_max_tokens[j_textCol][0] # count of encoded cols for j_textCol
                
                # get DistilBert last hidden state
                text = array[i][j]
                DistilBert_output = encodeText(text, DistilBert_tokenizer, DistilBert_model)
                
                # append encoded text
                for k in range(len(DistilBert_output)):
                    thisRow.append(DistilBert_output[k])

                # fill empty places before encoded text
                for k in range(j_encodedCols - len(DistilBert_output)):
                    thisRow.insert(currentThisRowLength, 0)

                # title
                if i == 0:
                    for k in range(j_encodedCols):
                        newTitle.append(str(j_textCol) + '_DistilBert_' + str(k))

        resultArray.append(thisRow)

    return (resultArray, newTitle)

# get first element of DistilBert output
def getDistilBertOutput(text, DistilBert_tokenizer, DistilBert_model):
    DistilBert_output = DistilBert_tokenizer(text, return_tensors='tf')
    DistilBert_output = np.array(DistilBert_output['input_ids'])[0] # shape: (N)
    
    return DistilBert_output

# encode text using DistilBert tokenizer and DistilBert model
def encodeText(text, DistilBert_tokenizer, DistilBert_model):
    regex = re.compile('[^a-zA-Z0-9.,?! ]')

    try:
        text = text.replace(r'\r\n', ' ')
        text = regex.sub('', text)
        
        DistilBert_output = getDistilBertOutput(text, DistilBert_tokenizer, DistilBert_model)
    except:
        DistilBert_output = np.array([0])
        
    return DistilBert_output

# find the max count of the number of DistilBert tokens for each column (project_title, ..., project_resource_summary)
def get_DistilBert_token_count(DistilBert_tokenizer, DistilBert_model):

    # read saved max tokens count
    try:
        DistilBert_max_tokens = pd.read_csv('DistilBert_max_tokens.csv', index_col=0)
        return np.array(DistilBert_max_tokens)

    # new max tokens count
    except:
        titles = ['project_title', 'project_essay_1', 'project_essay_2',
                  'project_essay_3', 'project_essay_4', 'project_resource_summary']
        
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        DistilBert_max_tokens = [0, 0, 0, 0, 0, 0]

        # for training data
        print('find DistilBert token count for training data ...')
        for i in range(len(train_data)):
            if i % 250 == 0: print(i)
            
            for j in range(6):
                DistilBert_max_tokens[j] = max(DistilBert_max_tokens[j],
                                               len(encodeText(train_data.iloc[i][titles[j]],
                                                              DistilBert_tokenizer, DistilBert_model)))

        # for test data
        print('find DistilBert token count for test data ...')
        for i in range(len(test_data)):
            if i % 250 == 0: print(i)
            
            for j in range(6):
                DistilBert_max_tokens[j] = max(DistilBert_max_tokens[j],
                                               len(encodeText(test_data.iloc[i][titles[j]],
                                                              DistilBert_tokenizer, DistilBert_model)))

        DistilBert_max_tokens_ = pd.DataFrame(np.array(DistilBert_max_tokens))
        DistilBert_max_tokens_.to_csv('DistilBert_max_tokens.csv')

        return np.array(DistilBert_max_tokens)

if __name__ == '__main__':

    train_option = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 0]
    test_option = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1]

    # GET DISTILBERT TOKEN COUNT
    DistilBert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    DistilBert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    DistilBert_max_tokens = get_DistilBert_token_count(DistilBert_tokenizer, DistilBert_model)

    # TRAINING
    
    # extract data
    train_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    (train_extracted, train_newTitle) = extract('train.csv', train_option, train_title,
                                                DistilBert_tokenizer, DistilBert_model, DistilBert_max_tokens)

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

    (test_extracted, test_newTitle) = extract('test.csv', test_option, test_title,
                                              DistilBert_tokenizer, DistilBert_model, DistilBert_max_tokens)

    test_toWrite = [] # [test_newTitle]
    
    for i in range(len(test_extracted)):
        test_toWrite.append(test_extracted[i])
        
    test_toWrite = pd.DataFrame(np.array(test_toWrite))
    test_toWrite.to_csv('test_input_text.csv')
