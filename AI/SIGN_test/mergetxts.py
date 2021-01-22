def mergeTextFiles(idList):

    result = ''
    
    for i in range(len(idList)):

        # open each info file
        try:
            f = open(str(idList[i]) + '-' + str(idList[i] + 1) + '_info.txt', 'r')
            fr = f.readlines()
            f.close()
        except:
            pass

        # write first line
        if i == 0: result += fr[0]

        # write info lines for each file
        for j in range(1, len(fr)): result += fr[j]

    # write the result
    fw = open('merged_info.txt', 'w')
    fw.write(result)
    fw.close()

if __name__ == '__main__':

    # for each 'X-(X+1)_info.txt' -> append X to idList
    idList = [330, 603, 948, 1000, 1231]
    
    mergeTextFiles(idList)
