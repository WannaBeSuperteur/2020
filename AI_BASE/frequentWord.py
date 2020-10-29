# dataFrame  : original dataframe
# specificCol: column -> columns that indicate the number of appearance of frequent words
def appearanceOfFrequentWordsTest(dataFrame, specificCol):

    numOfItems = 500 # number of items
    print(dataFrame)

    # find all words from specificCol (for first 100 items)
    allWords = ''
    for i in range(numOfItems):
        allWords += str(dataFrame.at[i, specificCol]) + '/'

    # make list for all words appeared in first 100 items
    allWords = allWords.lower()
    allWords = re.sub('[^a-z ]+', ' ', allWords)
    allWordsList = list(set(allWords.split(' ')))
    for i in range(len(allWordsList)): allWordsList[i] = (allWordsList[i], 0)
    allWordsDict = dict(allWordsList)
    allWordsDict.pop('') # remove blank

    # count appearance (make dictionary)
    for i in range(numOfItems):
        words = str(dataFrame.at[i, specificCol])
        words = words.lower()
        words = re.sub('[^a-z ]+', ' ', words)
        wordsList = list(set(words.split(' ')))

        # remove blank from wordsList
        try:
            wordsList.remove('')
        except:
            doNothing = 0 # do nothing

        # add 1 for each appearance of the word
        for j in range(len(wordsList)): allWordsDict[wordsList[j]] += 1

    # sort the array
    allWordsDict = sorted(allWordsDict.items(), key=(lambda x:x[1]), reverse=True)
    
    print('\n\n======== TEST RESULT ========\n')
    print(allWordsDict)
    print('\n=============================\n')

# dataFrame    : original dataframe
# specificCol  : column -> columns that indicate the number of appearance of frequent words
# frequentWords: list of frequent words
def appearanceOfFrequentWords(dataFrame, specificCol, frequentWords):

    # if one or more of specificCol or frequentWords are None, return initial dataFrame
    if specificCol == None or frequentWords == None: return dataFrame

    # add column that indicates each frequence word appearance
    for i in range(len(frequentWords)): # for each frequent word
        colName = 'CT_' + frequentWords[i] # number of column

        result = [] # check if each data include the word (then 1, else 0)
        for j in range(len(dataFrame)):
            if frequentWords[i] in dataFrame.at[j, specificCol]: result.append(1)
            else: result.append(0)

        dataFrame[colName] = result # add this column to dataframe

    # remove specificCol from the dataFrame
    dataFrame = dataFrame.drop([specificCol], axis='columns')

    return dataFrame
