import os

def mergeTextFiles(idList):

    result = ''
    
    for i in range(len(idList)):
        if i % 10 == 0: print(i)

        # open each info file
        try:
            f = open(('%06d' % idList[i]) + '_info.txt', 'r')
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

    # extract the list of ID
    files = os.listdir()
    idList = []

    for i in range(len(files)):
        nameSplit = files[i].split('_')

        try:
            if nameSplit[1] == 'info.txt':
                idList.append(int(nameSplit[0]))
        except:
            pass

    # for each 'X-(X+1)_info.txt' -> append X to idList
    mergeTextFiles(idList)
