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
#            2 -> binary
#            3 -> multiple one-hot
#            4 -> length
#            5 -> continuous with converted value
def convertToText(fn, columns, options, onehots, avgs, stddevs):

    txtFn = fn[:-4] + '.txt'
    data = pd.DataFrame(pd.read_csv(fn))
    rows = len(data)
    cols = len(columns)
    result = []
    false_val = -1

    print(data)
    for i in range(cols):
        print(data[columns[i]])

    # convert for option 4
    # length
    for i in range(cols):
        if options[i] == 4:
            for j in range(rows):
                try:
                    data.at[j, columns[i]] = len(data.at[j, columns[i]])
                except:
                    data.at[j, columns[i]] = 0

    # convert for option 5
    # yyyy-mm-dd to numeric value
    for i in range(cols):
        if options[i] == 5:
            for j in range(rows):

                ymd = data.at[j, columns[i]]
                y = int(ymd.split('-')[0])
                m = int(ymd.split('-')[1])
                d = int(ymd.split('-')[2])
                
                data.at[j, columns[i]] = (y - 1900) * 372 + m * 31 + d

    # one-hot for option 0 or 3
    if onehots == None:
        onehots = []

        for i in range(cols):
            if options[i] == 0:
                
                onehots.append(list(set(data[columns[i]])))
                
            elif options[i] == 3:

                # for example, list ['a', 'b', 'c']
                unique_items = []

                # for example, text ['a', 'b', 'c', 'a', 'a']
                items = data[columns[i]]
                
                for j in range(len(items)):

                    # for example, list ["'a'", "'b'", "'c'", "'a'", "'a'"]
                    item = items[j].split('[')[1].split(']')[0].split(', ')

                    # for example, list ['a', 'b', 'c', 'a', 'a']
                    item = [each_item[1:len(each_item)-1] for each_item in item]

                    # append each item 'a', 'b' and 'c' to unique_items
                    for k in range(len(item)):
                        if item[k] != '':
                            unique_items.append(item[k])

                onehots.append(list(set(unique_items)))
                
            else:
                onehots.append(None)

    # average and stddev for option 1, 4 or 5
    if avgs == None or stddevs == None:
        avgs = []
        stddevs = []

        for i in range(cols):
            if options[i] == 1 or options[i] == 4 or options[i] == 5:
                avgs.append(np.mean(data[columns[i]]))
                stddevs.append(np.std(data[columns[i]]))
            else:
                avgs.append(None)
                stddevs.append(None)

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

            # 1, 4 or 5 -> normalize using avg and stddev
            elif options[j] == 1 or options[j] == 4 or options[j] == 5:
                thisRow.append((float(cell) - avgs[j]) / stddevs[j])

            # 2 -> binary
            elif options[j] == 2:
                if cell == 1 or cell == 'True' or cell == 'TRUE':
                    thisRow.append(1)
                else:
                    thisRow.append(false_val)

            # 3 -> multiple one-hot
            elif options[j] == 3:
                for k in range(len(onehots[j])):
                    if "'" + onehots[j][k] + "'" in cell:
                        thisRow.append(1) 
                    else:
                        thisRow.append(false_val)

        result.append(thisRow)

    print(onehots)
    print(avgs)
    print(stddevs)

    # save the file
    RD.saveArray(txtFn, result, '\t', 500)

    return (onehots, avgs, stddevs)

def writeFile(name, columns, options):

    # training data
    (onehots, avgs, stddevs) = convertToText('yelp_training_set_' + name + '.csv',
                                             columns, options, None, None, None)

    # test data
    (_, _, _) = convertToText('yelp_test_set_' + name + '.csv',
                               columns, options, onehots, avgs, stddevs)

# write final training / test input
# trainTest : 'training' or 'test'
# rows      : row count for [business, checkin, review, user]
def writeFinalInput(trainTest, rows):

    business = RD.loadArray('yelp_' + str(trainTest) + '_set_business.txt')
    checkin = RD.loadArray('yelp_' + str(trainTest) + '_set_checkin.txt')
    review = RD.loadArray('yelp_' + str(trainTest) + '_set_review.txt')
    user = RD.loadArray('yelp_' + str(trainTest) + '_set_user.txt')

    finalInput = []
    
    for i in range(rows[2]):
        if i % 500 == 0: print('row : ' + str(i))
        
        thisRow = []

        # append review info
        # columns : ['user_id', 'business_id', 'text', 'stars', 'date']
        for j in range(2, 5):
            thisRow.append(review[i][j])

        # append user info
        # columns : ['user_id', 'review_count', 'average_stars']
        userid = review[i][0]
        users = rows[3]
        userFound = False

        for j in range(users):
            if user[j][0] == userid:
                for k in range(1, 3):
                    thisRow.append(user[j][k])
                userFound = True

        if userFound == False:
            for k in range(2): thisRow.append(0)

        # append business info
        # columns : ['business_id', 'review_count', 'longitude', 'stars', 'latitude', 'open']
        businessid = review[i][1]
        businesses = rows[0]
        businessFound = False

        for j in range(businesses):
            if business[j][0] == businessid:
                for k in range(1, 6):
                    thisRow.append(business[j][k])
                businessFound = True

        if businessFound == False:
            for k in range(5): thisRow.append(0)

        # append checkin info
        # columns : ['business_id', 'checkin_info']
        checkins = rows[1]
        checkinFound = False

        for j in range(checkins):
            if checkin[j][0] == businessid:
                thisRow.append(checkin[j][1])
                checkinFound = True

        if checkinFound == False:
            thisRow.append(0)

        finalInput.append(thisRow)

    if trainTest == 'training':
        RD.saveArray('train_input.txt', finalInput)
    elif trainTest == 'test':
        RD.saveArray('test_input.txt', finalInput)

# write final training output for USEFUL votes
def writeFinalOutput(reviews):

    train_review = np.array(pd.read_csv('yelp_training_set_review.csv'))
    train_output = []

    for i in range(reviews):

        # for example, 'funny': 0, 'useful': 0, 'cool': 0
        votes = train_review[i][1].split('{')[1].split('}')[0]
        useful = int(votes.split(',')[1].split(': ')[1])
        
        train_output.append(useful)

    # average and stddev of each array TO_xxxx
    avg = np.mean(train_output)
    stddev = np.std(train_output)

    # normalize each TO_xxxx
    for i in range(reviews):
        train_output[i] = [(train_output[i] - avg) / stddev]

    # save
    RD.saveArray('train_output.txt', train_output)
    RD.saveArray('train_output_avg_and_std.txt', [[avg, stddev]])

if __name__ == '__main__':

    # SUBMISSION
    # vote of reviews

    # DATASET INFO
    # (training count)
    # (test count)
    #
    # business : city, review_count, name, neighborhood, type, [ business_id ], full_address,
    # (11537)    state, longitude, stars, latitude, open, categories
    # (1205)     --------
    #            review_count  -> continuous
    #            longitude     -> continuous
    #            stars         -> continuous
    #            latitude      -> continuous
    #            open          -> binary
    #
    # checkin  : checkin_info, type, [ business_id ]
    # (8282)     --------
    # (734)      checkin_info  -> length
    #
    # review   : [ user_id ], review_id, text, [ business_id ], stars, date, type
    # (229907)   --------
    # (22956)    text          -> length
    #            stars         -> continuous
    #            date          -> continuous with converted value
    #
    # user     : review_count, name, average_stars, [ user_id ], type
    # (43873)    --------
    # (5105)     review_count  -> continuous
    #            average_stars -> continuous

    # FINAL INPUT
    #                 : (review)   text          -> length
    #                 : (review)   stars         -> continuous
    #                 : (review)   date          -> continuous with converted value
    # [ user_id ]     : (user)     review_count  -> continuous
    # [ user_id ]     : (user)     average_stars -> continuous
    # [ business_id ] : (business) review_count  -> continuous
    # [ business_id ] : (business) longitude     -> continuous
    # [ business_id ] : (business) stars         -> continuous
    # [ business_id ] : (business) latitude      -> continuous
    # [ business_id ] : (business) open          -> binary
    # [ business_id ] : (checkin)  checkin_info  -> length

    # FINAL OUTPUT
    #                 : (review)   votes         -> (useful) only

    # rows: business, checkin, review, user
    train_rows = [11537, 8282, 229907, 43873]
    test_rows = [1205, 734, 22956, 5105]

    # write info files
    writeFile('business', ['business_id', 'review_count', 'longitude', 'stars', 'latitude', 'open'], [-1, 1, 1, 1, 1, 2])
    writeFile('checkin', ['business_id', 'checkin_info'], [-1, 4])
    writeFile('review', ['user_id', 'business_id', 'text', 'stars', 'date'], [-1, -1, 4, 1, 5])
    writeFile('user', ['user_id', 'review_count', 'average_stars'], [-1, 1, 1])

    # write final training input
    writeFinalInput('training', train_rows)

    # write final training output
    writeFinalOutput(train_rows[2])

    # write final test input
    writeFinalInput('test', test_rows)
