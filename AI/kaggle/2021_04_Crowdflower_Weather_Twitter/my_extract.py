import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import pandas as pd
import re

# write input data
def writeInput(array_input, state_list, modify_state_list):

    # transform state_list: ['alabama', 'alaska', ...]
    #                  into [['alabama', 'AL'], ['alaska', 'AK'], ...]
    # because 'NYC' includes 'NY', no need to find 'NYC' for New York
    if modify_state_list == True:
        state_set = {'alabama':'AL', 'alaska':'AK', 'arizona':'AZ',
                     'arkansas':'AR', 'california':'CA', 'colorado':'CO',
                     'connecticut':'CT', 'delaware':'DE', 'florida':'FL',
                     'georgia':'GA', 'hawaii':'HI', 'idaho':'ID',
                     'illinois':'IL', 'indiana':'IN', 'iowa':'IA',
                     'kansas':'KS', 'kentucky':'KY', 'louisiana':'LA',
                     'maine':'ME', 'maryland':'MD', 'massachusetts':'MA',
                     'michigan':'MI', 'minnesota':'MN', 'mississippi':'MS',
                     'missouri':'MO', 'montana':'MT', 'nebraska':'NE',
                     'nevada':'NV', 'new hampshire':'NH', 'new jersey':'NJ',
                     'new mexico':'NM', 'new york':'NY', 'north carolina':'NC',
                     'north dakota':'ND', 'ohio':'OH', 'oklahoma':'OK',
                     'oregon':'OR', 'pennsylvania':'PA', 'rhode island':'RI',
                     'south carolina':'SC', 'south dakota':'SD', 'tennessee':'TN',
                     'texas':'TX', 'utah':'UT', 'vermont':'VT',
                     'virginia':'VA', 'washington':'WA', 'west virginia':'WV',
                     'wisconsin':'WI', 'wyoming':'WY', 'district of columbia':'DC'}
        
        for i in range(len(state_list)):
            state_list[i] = [state_list[i], state_set[state_list[i]]]

    # write input
    final_input = []

    for i in range(len(array_input)):
        thisRow = []

        # tweet (remove alphabet and space only)
        thisRow.append(re.sub('[^a-zA-Z ]', '', array_input[i][0]))

        # state or location
        for j in range(len(state_list)):

            # state name exist
            if state_list[j][0] == array_input[i][1]:
                thisRow.append(1)

            # state symbol (AL, AK, ...) exist
            elif array_input[i][2] != None:
                try:
                    if state_list[j][1] in array_input[i][2]:
                        thisRow.append(1)
                    else:
                        thisRow.append(falseVal)
                except:
                    thisRow.append(falseVal)
                    
            else:
                thisRow.append(falseVal)

        final_input.append(thisRow)

    return final_input

if __name__ == '__main__':

    # configuration
    falseVal = -1

    # create train input and output
    train_data = np.array(pd.read_csv('train.csv'))
    train_input = train_data[:, 1:4]
    train_output = train_data[:, 4:]

    test_data = np.array(pd.read_csv('test.csv'))
    test_input = test_data[:, 1:4]

    # one-hot for state
    state_list = list(set(train_data[:, 2]))

    # write final train_input and test_input
    # test output is as original
    final_train_input = writeInput(train_input, state_list, True)
    final_test_input = writeInput(test_input, state_list, False)

    # save training and test data
    RD.saveArray('train_input.txt', final_train_input, '\t', 500)
    RD.saveArray('train_output.txt', train_output, '\t', 500)
    RD.saveArray('test_input.txt', final_test_input, '\t', 500)

    # split training input into train_train and train_valid
    validRate = 0.15
    trainRows = len(final_train_input)

    train_train_input = []
    train_train_output = []
    train_valid_input = []
    train_valid_output = []

    for i in range(trainRows):
        if i >= validRate * trainRows: # train_train
            train_train_input.append(final_train_input[i])
            train_train_output.append(train_output[i])
            
        else: # train_valid
            train_valid_input.append(final_train_input[i])
            train_valid_output.append(train_output[i])

    # save splitted training data
    RD.saveArray('train_train_input.txt', train_train_input, '\t', 500)
    RD.saveArray('train_train_output.txt', train_train_output, '\t', 500)
    RD.saveArray('train_valid_input.txt', train_valid_input, '\t', 500)
    RD.saveArray('train_valid_output.txt', train_valid_output, '\t', 500)
