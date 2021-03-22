import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# word count for top 100 words for word count (marked as 6)
def getWordCount(option, trainFile):
    wordCount = []

    # use training data only
    train_array = np.array(pd.read_csv(trainFile))
    rows = len(train_array)
    cols = len(train_array[0])
    
    for i in range(cols):
        
        if option[i] == 6:

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
#  6 : word count (top 100 words)
def extract(fn, option, title, wordCount):

    array = np.array(pd.read_csv(fn))
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
            elif option[j] == 5:
                array[i][j] = len(str(array[i][j]))

    # avg and stddev for each column marked as 1, 4 or 5
    avgs = []
    stddevs = []
    
    for i in range(cols):
        if option[i] == 1 or option[i] == 4 or option[i] == 5:
            thisCol = array[:, i]
            avgs.append(np.mean(thisCol))
            stddevs.append(np.std(thisCol))
        else:
            avgs.append(-1)
            stddevs.append(-1)
            
    print(avgs)
    print(stddevs)

    # one-hot for each column marked as 2 or 3
    # None  for columns not marked as 2 or 3
    # array for columns     marked as 2 or 3
    # for example: [None, None, ['A', 'B', 'C', 'F'], ['math', 'science', 'computer'], None]
    print('\n ==== one-hot elements ====\n')
    
    onehot = []
    for i in range(cols):
        if option[i] == 2 or option[i] == 3:
            thisCol = list(array[:, i])
            thisCol_set = list(set(thisCol))
            onehot.append(thisCol_set)

            print(title[i] + ' : ' + str(len(thisCol_set)))
        else:
            onehot.append(None)

    print('\n ==========================\n')

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
                thisRow.append((float(array[i][j]) - avgs[j]) / stddevs[j])

                if i == 0: newTitle.append(str(j) + '_zval')

            #  2. one-hot vector
            #  3. one-hot vector with first 3 letters
            elif option[j] == 2 or option[j] == 3:
                for k in range(len(onehot[j])):
                    if array[i][j] == onehot[j][k]:
                        thisRow.append(1)
                    else:
                        thisRow.append(0)

                    if i == 0: newTitle.append(str(j) + '_hot_' + str(k))

            #  6. word count
            elif option[j] == 6:
                WC = wordCount[j]

                temp = str(array[i][j]).replace('.', '').replace(',', '').replace('?', '').replace('!', '')
                temp = temp.lower()
                temp = temp.split(' ')

                # avg and std for each j
                wc_avgs = [None, None, None, None, None, None, None, None, 1, 1, 1, 1, 1, 1, None, None]
                wc_stds = [None, None, None, None, None, None, None, None, 1, 1, 1, 1, 1, 1, None, None]

                for k in range(100):
                    count = 0
                    
                    for m in range(len(temp)):

                        # except for 'my' 'student' 'needs' 'to'
                        if temp[m] == WC[4 + k][0]: count += 1

                    thisRow.append((count - wc_avgs[j]) / wc_stds[j])

                    if i == 0: newTitle.append(str(j) + '_word_' + str(k))

        resultArray.append(thisRow)

    return (resultArray, newTitle)

if __name__ == '__main__':

    # GET WORD COUNT

    train_option = [-1, -1, 2, -1, -1, 2, 3, 3, 6, 6, 6, 6, 6, 6, 1, 0]
    test_option = [-1, -1, 2, -1, -1, 2, 3, 3, 6, 6, 6, 6, 6, 6, 1]
    
    wordCount = getWordCount(train_option, 'train.csv')

    # TRAINING
    
    # extract data
    train_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'project_is_approved']

    (train_extracted, train_newTitle) = extract('train.csv', train_option, train_title, wordCount)

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
        
    RD.saveArray('train_input.txt', train_input, '\t', 500)
    RD.saveArray('train_output.txt', train_output, '\t', 500)

    # TEST
    
    # extract data
    test_title = ['id', 'teacher_id', 'teacher_prefix', 'school_state',
             'project_submitted_datetime', 'project_grade_category',
             'project_subject_categories', 'project_subject_subcategories',
             'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary', 'teacher_number_of_previously_posted_projects']

    (test_extracted, test_newTitle) = extract('test.csv', test_option, test_title, wordCount)

    test_toWrite = [] # [test_newTitle]
    
    for i in range(len(test_extracted)):
        test_toWrite.append(test_extracted[i])
        
    RD.saveArray('test_input.txt', test_toWrite, '\t', 500)
