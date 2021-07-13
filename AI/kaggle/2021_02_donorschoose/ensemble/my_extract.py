import sys
sys.path.insert(0, '../../../AI_BASE')
import readData as RD
import random
import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# word count for top 25 words for word count (marked as 6)
def getWordCount(option, trainFile):
    wordCount = []

    # use training data only
    train_array = np.array(pd.read_csv(trainFile))
    rows = len(train_array)
    cols = len(train_array[0])
    
    for i in range(cols):
        
        if option[i] == 6 or option[i] == 56:

            # make the set of words
            setOfWords = {}
            
            for j in range(rows):
                temp = str(train_array[j][i]).replace('.', '').replace(',', '').replace('?', '').replace('!', '')
                temp = temp.lower()
                temp = temp.split(' ')
                
                for k in range(len(temp)):
                    try:
                        setOfWords[temp[k]] += 1
                    except:
                        setOfWords[temp[k]] = 1

            # make the list of words
            listOfWords = []

            for key, value in setOfWords.items():
                listOfWords.append([key, value])

            # sort by values
            listOfWords.sort(key=lambda x:x[1], reverse=True)

            wordCount.append(listOfWords)

        else:
            wordCount.append(None)

    return wordCount

# option: [op0, op1, ..., opN] for each column
# each option is:
# -1 : except for this column
#  0 : numeric
#  1 : numeric with normalization
#  2 : one-hot vector
#  3 : one-hot vector with first 3 letters
#  4 : yyyy-mm-dd hh:mm:ss to numeric with normalization
#  5 : length with normalization
def appendNormalizedLength(thisRow, textLen, avgLen, stdLen):
    thisRow.append((float(textLen) - avgLen) / stdLen)
    
#  6 : word count (top 25 words)
def appendWordCount(thisRow, wordCount, text, colIndex):

    temp = str(text).replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    temp = temp.lower()
    temp = temp.split(' ')

    # avg and std for each j
    wc_avgs = [None, None, None, None, None, None, None, None, 1, 1, 1, 1, 1, 1, None, None]
    wc_stds = [None, None, None, None, None, None, None, None, 1, 1, 1, 1, 1, 1, None, None]

    for k in range(25):
        count = 0
                    
        for m in range(len(temp)):

            # except for 'my' 'student' 'needs' 'to'
            if temp[m] == wordCount[4 + k][0]: count += 1

        thisRow.append((count - wc_avgs[colIndex]) / wc_stds[colIndex])
                    
#  7 : one-hot vector for yyyy-mm-dd hh:mm:ss -> year, month, DOW and hh

# onehot: saved one-hot vectors
# years : set of years (for example, [2007, 2008, 2009, 2010])
def extract(fn, option, title, wordCount, onehot, years):

    array = np.array(pd.read_csv(fn))
    array_textLen = np.array(pd.read_csv(fn))
    print(array)
    
    rows = len(array)
    cols = len(array[0])
    print(rows, cols)
    
    resultArray = []

    # convert yyyy-mm-dd hh:mm:ss into numeric for each column marked as 4
    for i in range(rows):
        if i % 2500 == 0: print(i)
        
        for j in range(cols):

            # leave leftmost 3 letters
            if option[j] == 3:
                array[i][j] = array[i][j][:3]
                
            # convert yyyy-mm-dd hh:mm:ss into numeric for each column marked as 4
            elif option[j] == 4:
                dateTime = array[i][j]

                date = dateTime.split(' ')[0]
                time = dateTime.split(' ')[1]

                year = date.split('-')[0]
                month = date.split('-')[1]
                day = date.split('-')[2]

                hour = time.split(':')[0]
                minute = time.split(':')[1]
                second = time.split(':')[2]

                array[i][j] = year * 31104000 + month * 2592000 + day * 86400 + hour * 3600 + minute * 60 + second

            # length with normalization
            elif option[j] == 5 or option[j] == 56:
                array_textLen[i][j] = len(str(array_textLen[i][j]))

    # avg and stddev for each column marked as 1, 4, 5 or 56
    avgs = []
    stddevs = []
    
    for i in range(cols):
        print('column ' + str(i))
        
        if option[i] == 1 or option[i] == 4 or option[i] == 5 or option[i] == 56:

            # text length (5 or 56)
            if option[i] == 5 or option[i] == 56:
                thisCol = array_textLen[:, i]

            # numeric value (1 or 4)
            else:
                thisCol = array[:, i]

            avgs.append(np.mean(thisCol))
            stddevs.append(np.std(thisCol))
        else:
            avgs.append(-1)
            stddevs.append(-1)
            
    print(avgs)
    print(stddevs)

    # one-hot for each column marked as 2, 3 or 8
    # None  for columns not marked as 2, 3 or 8
    # array for columns     marked as 2, 3 or 8
    # for example: [None, None, ['A', 'B', 'C', 'F'], ['math', 'science', 'computer'], None]
    if onehot == None:
        print('\n ==== one-hot elements ====\n')
        
        onehot = []
        for i in range(cols):
            if option[i] == 2 or option[i] == 3 or option[i] == 8:

                thisCol = list(array[:, i])

                # 2 or 3 -> original
                if option[i] == 2 or option[i] == 3:
                    thisCol_set = list(set(thisCol)) 

                # 8 -> split with delimiter ',' (eg: ['a,b', 'c', 'a'] -> ['a', 'b', 'c'])
                else:
                    thisCol_ = []

                    for j in range(len(thisCol)):
                        elements = thisCol[j].replace(' ', '').split(',')
                        for k in range(len(elements)):
                            thisCol_.append(elements[k])

                    thisCol_set = list(set(thisCol_))

                onehot.append(thisCol_set)
                print(title[i] + ' : ' + str(len(thisCol_set)))
            else:
                onehot.append(None)

        print('\n ==========================\n')

    else:
        print('\n=================================')
        print('  One-hot array already exists.')
        print('=================================\n')

    # for each row
    newTitle = []
    for i in range(rows):
        if i % 500 == 0: print(i)
        
        thisRow = []

        # for each column
        for j in range(cols):

            # -1 : except for this column
            if option[j] == -1:
                doNothing = 0

            #  0 : numeric
            elif option[j] == 0:
                thisRow.append(float(array[i][j]))

                if i == 0: newTitle.append(str(j) + '_val')

            #  1 : numeric with normalization
            #  4 : yyyy-mm-dd hh:mm:ss to numeric with normalization
            #  5 : length with normalization
            elif option[j] == 1 or option[j] == 4 or option[j] == 5:
                appendNormalizedLength(thisRow, array_textLen[i][j], avgs[j], stddevs[j])

                if i == 0: newTitle.append(str(j) + '_zval')

            #  2 : one-hot vector
            #  3 : one-hot vector with first 3 letters
            #  8 : one-hot vector with delimeter ','
            elif option[j] == 2 or option[j] == 3 or option[j] == 8:
                
                for k in range(len(onehot[j])):

                    # 2 or 3 -> original
                    if option[j] == 2 or option[j] == 3:
                        if array[i][j] == onehot[j][k]:
                            thisRow.append(1)
                        else:
                            thisRow.append(0)

                    # 8 -> split with delimiter ','
                    else:
                        if onehot[j][k].replace(' ', '') in array[i][j].replace(' ', ''):
                            thisRow.append(1)
                        else:
                            thisRow.append(0)

                    if i == 0: newTitle.append(str(j) + '_hot_' + str(k))

            #  6 : word count (TOP 25 words)
            elif option[j] == 6:
                
                appendWordCount(thisRow, wordCount[j], array[i][j])

                if i == 0:
                    for k in range(25):
                        newTitle.append(str(j) + '_word_' + str(k))

            #  7 : yyyy-mm-dd hh:mm:ss -> yyyy, mm, DOW and hh
            elif option[j] == 7:

                # split dateTime
                dateTime = array[i][j]

                date = dateTime.split(' ')[0]
                time = dateTime.split(' ')[1]

                year = date.split('-')[0]
                month = date.split('-')[1]
                day = date.split('-')[2]

                hour = time.split(':')[0]
                
                # vectors of year, month, DOW and hour
                # year : 2016 or 2017
                for k in range(len(years)):
                    newTitle.append(str(j) + '_year_' + str(years[k]))
                    
                    if int(year) == years[k]:
                        thisRow.append(1)
                    else:
                        thisRow.append(0)

                # month : 01 ~ 12
                for k in range(12):
                    newTitle.append(str(j) + '_month_' + str(k+1))
                    
                    if int(month) == k+1:
                        thisRow.append(1)
                    else:
                        thisRow.append(0)

                # DOW : 0 ~ 6 (Monday=0, Tuesday=1, ..., Sunday=6 with weekday() of datetime)
                DOW = datetime.datetime(int(year), int(month), int(day)).weekday()
                
                for k in range(7):
                    newTitle.append(str(j) + '_dow_' + str(k))
                    
                    if DOW == k:
                        thisRow.append(1)
                    else:
                        thisRow.append(0)

                # hour : 00 ~ 23
                for k in range(24):
                    newTitle.append(str(j) + '_hour_' + str(k))
                    
                    if int(hour) == k:
                        thisRow.append(1)
                    else:
                        thisRow.append(0)

            # 56 : length with normalization + word count
            elif option[j] == 56:
                appendWordCount(thisRow, wordCount[j], array[i][j], j)
                appendNormalizedLength(thisRow, array_textLen[i][j], avgs[j], stddevs[j])

                if i == 0:
                    for k in range(25):
                        newTitle.append(str(j) + '_word_' + str(k))

        resultArray.append(thisRow)

    return (resultArray, newTitle, onehot)

if __name__ == '__main__':

    # GET WORD COUNT

    train_option = [-1, -1, 2, -1, -1, 2, 8, 8, 56, 56, 56, 56, 56, 56, 1, 0]
    test_option = [-1, -1, 2, -1, -1, 2, 8, 8, 56, 56, 56, 56, 56, 56, 1]
    
    wordCount = getWordCount(train_option, '../train.csv')

    # TRAINING
    
    # extract data
    train_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    (train_extracted, train_newTitle, onehot) = extract('../train.csv', train_option, train_title, wordCount, None, [2016, 2017])

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
        
    pd.DataFrame(train_input).to_csv('train_input.csv')
    pd.DataFrame(train_output).to_csv('train_output.csv')

    # TEST
    
    # extract data
    test_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects']

    (test_extracted, test_newTitle, _) = extract('../test.csv', test_option, test_title, wordCount, onehot, [2016, 2017])

    test_toWrite = [] # [test_newTitle]
    
    for i in range(len(test_extracted)):
        test_toWrite.append(test_extracted[i])
        
    pd.DataFrame(test_toWrite).to_csv('test_input.csv')
