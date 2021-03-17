import math
import numpy as np
import pandas as pd
import random

# train.csv -> train_train.csv, train_valid.csv

# designate rows to valid
def designateValidRows(n, valid_rate):

    valid = []
    for i in range(n): valid.append(False)

    valid_count = 0

    while valid_count < n * valid_rate:
        row_to_valid = random.randint(0, n-1)
        
        if valid[row_to_valid] == False:
            valid[row_to_valid] = True
            valid_count += 1

    return valid

if __name__ == '__main__':

    valid_rate = 0.05

    # load training data
    train = np.array(pd.read_csv('train.csv', dtype={11:str, 12:str}))
    rows = len(train)

    # designate rows to valid
    rows_to_valid = designateValidRows(rows, valid_rate)

    # create training and valid data
    training = []
    validation = []
    
    for i in range(rows):
        if rows_to_valid[i] == True:
            validation.append(train[i][1:])
        else:
            training.append(train[i][1:])

    # save
    print('writing train_train.csv ...')
    pd.DataFrame(training).to_csv('train_train.csv', sep=',')
    
    print('writing train_valid.csv ...')
    pd.DataFrame(validation).to_csv('train_valid.csv', sep=',')
