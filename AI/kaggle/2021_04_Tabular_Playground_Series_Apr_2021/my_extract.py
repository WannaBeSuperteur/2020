import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import pandas as pd
import random
import re

# fn      : name of csv file
# columns : names of columns to exploit
# options : -1 -> original
#            0 -> one-hot
#            1 -> continuous
#            2 -> log(continuous)
# mdopts  : missing data options
#            0 -> zero
#            1 -> average
def convertToTextFile(fn, columns, options, mdopts, onehots, avgs, stddevs, valid, isValid):

    if valid == True:
        txtFn0 = fn[:-4] + '_train_input.txt'
    else:
        txtFn0 = fn[:-4] + '_input.txt'
        
    txtFn1 = fn[:-4] + '_valid_input.txt'
    result_train = []
    result_valid = []
        
    data = pd.DataFrame(pd.read_csv(fn))
    rows = len(data)
    cols = len(columns)
    false_val = -1

    # make the valid array
    if valid == False:
        isValid = [False]*rows

    assert(rows == len(isValid))

    print(data)
    for i in range(cols):
        print(data[columns[i]])

    # average and stddev for option 1 or 2
    if avgs == None or stddevs == None:
        avgs = []
        stddevs = []

        for i in range(cols):
            if options[i] == 1:
                avgs.append(np.mean(data[columns[i]]))
                stddevs.append(np.std(data[columns[i]]))
            elif options[i] == 2:
                avgs.append(np.mean(np.log(data[columns[i]] + 1.0)))
                stddevs.append(np.std(np.log(data[columns[i]] + 1.0)))
            else:
                avgs.append(None)
                stddevs.append(None)

    # filling missing values
    for i in range(cols):
        for j in range(rows):

            if pd.isnull(data.at[j, columns[i]]):

                # zero
                if mdopts[i] == 0:
                    data.at[j, columns[i]] = 0

                # average
                elif mdopts[i] == 1:
                    data.at[j, columns[i]] = avgs[i]

    # one-hot for option 0
    if onehots == None:
        onehots = []

        for i in range(cols):
            if options[i] == 0:   
                onehots.append(list(set(data[columns[i]])))
            else:
                onehots.append(None)

    # append to result
    for i in range(rows):
        if i % 1000 == 0: print(i)
        
        thisRow = []
        
        for j in range(cols):

            cell = data.at[i, columns[j]]

            # -1 -> original
            if options[j] == -1:
                thisRow.append(cell)

            # 0 -> one-hot
            elif options[j] == 0:
                for k in range(len(onehots[j])):
                    if onehots[j][k] == cell:
                        thisRow.append(1)
                    else:
                        thisRow.append(false_val)

            # 1 or 2 -> normalize using avg and stddev
            elif options[j] == 1:
                thisRow.append((float(cell) - avgs[j]) / stddevs[j])
            elif options[j] == 2:
                thisRow.append((np.log(float(cell) + 1.0) - avgs[j]) / stddevs[j])

        # reference: https://www.kaggle.com/jeongyoonlee/dae-with-2-lines-of-code-with-kaggler

        # family size
        familySize = data.at[i, 'SibSp'] + data.at[i, 'Parch']
        thisRow.append(familySize - 1)

        # ticket No.
        ticket = str(data.at[i, 'Ticket'])
        ticket_alp = re.sub('[^A-Z]', '', ticket)
        ticket_num = re.sub('[^0-9]', '', ticket)

        # ticket No. -> alphabet part
        ticket_alp_map = {'CA':0, 'A':1, 'AS':2, 'PC':3, 'WC':4, '':5}
        toAdd_t = [false_val, false_val, false_val, false_val, false_val,
                   false_val, false_val]
        try:
            toAdd_t[ticket_alp_map[ticket_alp]] = 1
        except:
            toAdd_t[6] = 1
        thisRow += toAdd_t

        # ticket No. -> number part
        try:
            ticket_num = int(ticket_num)
                
            if ticket_num < 100000:
                thisRow += [1, false_val, false_val, false_val]
            elif ticket_num < 1000000:
                thisRow += [false_val, 1, false_val, false_val]
            else:
                thisRow += [false_val, false_val, 1, false_val]
        except:
            thisRow += [false_val, false_val, false_val, 1]

        # cabin No.
        cabin = str(data.at[i, 'Cabin'])
        cabin_alp = cabin[0]

        # cabin No. -> alphabet part
        cabin_alp_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'X':8}
        toAdd_c = [false_val, false_val, false_val, false_val, false_val,
                   false_val, false_val, false_val, false_val, false_val]
        try:
            toAdd_c[cabin_alp_map[cabin_alp]] = 1
        except:
            toAdd_c[9] = 1
        thisRow += toAdd_c

        # name -> First Name, Last Name -> first letter for each name
        name = str(data.at[i, 'Name'])
        firstName = name.split(', ')[0]
        lastName = name.split(', ')[1]
        firstNameInitial = firstName[0]
        lastNameInitial = lastName[0]

        # mapping : letter -> numeric value
        mapping = {'A':-1.2, 'B':-1.1, 'C':-1.0, 'D':-0.9, 'E':-0.8,
                   'F':-0.7, 'G':-0.6, 'H':-0.5, 'I':-0.4, 'J':-0.3,
                   'K':-0.2, 'L':-0.1, 'M':0.0, 'N':0.1, 'O':0.2,
                   'P':0.3, 'Q':0.4, 'R':0.5, 'S':0.6, 'T':0.7,
                   'U':0.8, 'V':0.9, 'W':1.0, 'X':1.1, 'Y':1.2,
                   'Z':1.3}

        thisRow.append(mapping[firstNameInitial])
        thisRow.append(mapping[lastNameInitial])

        # finally append
        if isValid[i] == True:
            result_valid.append(thisRow)
        else:
            result_train.append(thisRow)

    print(onehots)
    print(avgs)
    print(stddevs)

    # save the file
    RD.saveArray(txtFn0, result_train, '\t', 500)
    
    if valid == True:
        RD.saveArray(txtFn1, result_valid, '\t', 500)

    return (onehots, avgs, stddevs)

def writeFile(columns, options, mdopts, valid, isValid):

    # training data
    (onehots, avgs, stddevs) = convertToTextFile('train.csv', columns, options, mdopts,
                                                 None, None, None,
                                                 valid, isValid)

    # test data
    (_, _, _) = convertToTextFile('test.csv', columns, options, mdopts,
                                  onehots, avgs, stddevs,
                                  False, False)

# write final training output for USEFUL votes
def writeOutput(valid, isValid):

    trainData = np.array(pd.read_csv('train.csv'))
    train_output = []

    if valid == True:
        valid_output = []

    print(trainData)

    for i in range(len(trainData)):
        if isValid[i] == True:
            valid_output.append([trainData[i][1]]) # index of 'Survived' is 1
        else:
            train_output.append([trainData[i][1]])

    # save
    if valid == True:
        RD.saveArray('train_train_output.txt', train_output)
        RD.saveArray('train_valid_output.txt', valid_output)
    else:
        RD.saveArray('train_output.txt', train_output)

if __name__ == '__main__':

    # SUBMISSION
    # prediction of 'survived' column

    # business : [ PassengerId ], Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket,
    # (100000)   Fare, Cabin, Embarked
    # (100000)

    # FINAL INPUT
    # Pclass        -> one-hot
    # Sex           -> one-hot
    # Age           -> continuous
    # SibSp         -> continuous
    # Parch         -> continuous
    # Fare          -> log(continuous)
    # Embarked      -> one-hot

    # FINAL OUTPUT
    # Survived      -> original

    # rows
    train_rows = 100000
    test_rows = 100000
    valid_rate = float(RD.loadArray('val_rate.txt')[0][0])

    isValid = []
    for i in range(train_rows):
        isValid.append(False)

    count = 0
    while count < train_rows * valid_rate:
        rand = random.randint(0, train_rows - 1)

        if isValid[rand] == False:
            count += 1
            isValid[rand] = True

    # write info files
    writeFile(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
              [0, 0, 1, 1, 1, 2, 0],
              [1, 1, 1, 1, 1, 1, 1],
              valid_rate > 0, isValid)
    
    # write training output
    writeOutput(valid_rate > 0, isValid)
