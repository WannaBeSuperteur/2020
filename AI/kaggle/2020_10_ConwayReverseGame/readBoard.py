import math
import numpy as np

# files read by this code
# train_input_sub_X.txt  (open in readBoard)
# train_output_sub_X.txt (open in readBoard)
# test_input_sub_X.txt   (open in readBoard)
# final_X.txt            (open in readBoard)

# files written by this code
# train_input_sub_X_read_result.txt  (open in readBoard)
# train_output_sub_X_read_result.txt (open in readBoard)
# test_input_sub_X_read_result.txt   (open in readBoard)
# final_X_read_result.txt            (open in readBoard)

def readBoard(range_, fileName, way, splitter, startCol):

    if range_ == None or fileName == None or way == None or splitter == None:
        print("""
            range    : A~B: read from A-th line to B-th line (including both end)
                       A~E: read from A-th line to end
                       S~B: read from start to B-th line
                       S~E: read all lines
            startCol : start index of column to print
            fileName : file to read
            way      : how to read files (0 or square)
            splitter : splitter ('Tab' for \t)
            """)

        range_ = input()
        fileName = input()
        way = input()
        splitter = input()
        startCol = input()

    # support Tab splitter
    if splitter == 'Tab': splitter = '\t'

    # read file
    f = open(fileName, 'r')
    fl = f.readlines()
    f.close()

    # get start and end
    start = range_.split('~')[0]
    end = range_.split('~')[1]
    
    if start == 'S': start_ = 0
    else: start_ = int(start)

    if end == 'E': end_ = len(fl)-1
    else: end_ = int(end)

    # write result
    result = ''
        
    for i in range(start_, end_+1):
        print(str(i) + ' / ' + str(start_) + ',' + str(end_))

        # this row
        readResult = np.array(fl[i].split('\n')[0].split(splitter))[startCol:]
        
        if way == 'square':
            sqrtVal = int(math.sqrt(len(readResult)))

            # convert to print form
            readResult = str(i) + '\n' + str(readResult.reshape(sqrtVal, sqrtVal)) + '\n'
            
            result += readResult

    f = open(fileName[:len(fileName)-4] + '_read_result.txt', 'w')
    f.write(result)
    f.close()

if __name__ == '__main__':
    np.set_printoptions(edgeitems=1000, linewidth=10000)

    for i in range(5):
        # readBoard('S~400', 'train_input_n_sub_' + str(i) + '.txt', 'square', 'Tab')
        # readBoard('S~400', 'train_output_n_sub_' + str(i) + '.txt', 'square', 'Tab')
        readBoard('S~9', 'train_input_sub_' + str(i) + '.txt', 'square', 'Tab', 0)
        readBoard('S~9', 'train_output_sub_' + str(i) + '.txt', 'square', 'Tab', 0)
        readBoard('S~9', 'test_input_sub_' + str(i) + '.txt', 'square', 'Tab', 0)
        readBoard('S~9', 'final_' + str(i) + '.csv', 'square', ',', 1)
