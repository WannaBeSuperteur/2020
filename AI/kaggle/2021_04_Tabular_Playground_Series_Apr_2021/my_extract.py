import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import pandas as pd

# fn      : name of csv file
# columns : names of columns to exploit
# options : -1 -> original
#            0 -> one-hot
#            1 -> continuous
#            2 -> log(continuous)
# mdopts  : missing data options
#            0 -> zero
#            1 -> average
def convertToTextFile(fn, columns, options, mdopts, onehots, avgs, stddevs):

    txtFn = fn[:-4] + '_input.txt'
    data = pd.DataFrame(pd.read_csv(fn))
    rows = len(data)
    cols = len(columns)
    result = []
    false_val = -1

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
    # yyyy-mm-dd to numeric value
    for i in range(rows):
        
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

        result.append(thisRow)

    print(onehots)
    print(avgs)
    print(stddevs)

    # save the file
    RD.saveArray(txtFn, result, '\t', 500)

    return (onehots, avgs, stddevs)

def writeFile(columns, options, mdopts):

    # training data
    (onehots, avgs, stddevs) = convertToTextFile('train.csv', columns, options, mdopts, None, None, None)

    # test data
    (_, _, _) = convertToTextFile('test.csv', columns, options, mdopts, onehots, avgs, stddevs)

# write final training output for USEFUL votes
def writeOutput():

    trainData = np.array(pd.read_csv('train.csv'))
    train_output = []

    print(trainData)

    for i in range(len(trainData)):
        train_output.append([trainData[i][1]]) # index of 'Survived' is 1

    # save
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

    # write info files
    writeFile(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
              [0, 0, 1, 1, 1, 2, 0],
              [1, 1, 1, 1, 1, 1, 1])
    
    # write training output
    writeOutput()
