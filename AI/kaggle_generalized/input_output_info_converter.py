# convert the contents in input_output_info_original.txt
# and save them to input_output_info_converted.txt
# For example,

# for training input/output and test input
# 0~9 -> 0 1 2 3 4 5 6 7 8 9
# 10~14t -> 10t 11t 12t 13t 14t
# 15~19z -> 15z 16z 17z 18z 19z
# 20~29d1 -> 20d1 21d1 22d1 23d1 24d1 25d1 26d1 27d1 28d1 29d1
# 36~39t -> 36t 37t 38t 39t
# 42~44z -> 42z 43z 44z

# for test output
# o2 -> o o
# r3 -> r r r
# o5 -> o o o o o

# lineSplit: splitted result of the line (for example, ['0~9', '10~14t', '15~19z'])
def convertLine(lineSplit):
    
    result = lineSplit[0] # converted result of training input/output, test input info

    # for each column number (including range form)
    for i in range(1, len(lineSplit)):
        
        if '~' in lineSplit[i]: # in the range form

            # check symbol
            symbol = ''
            if 't' in lineSplit[i]: # text
                symbol = 't'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-1]
            elif 'd1' in lineSplit[i]: # date type mm/dd/yyyy (converted to Z value)
                symbol = 'd1'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-2]
            elif 'd' in lineSplit[i]: # date type yyyy-mm-dd (converted to Z value)
                symbol = 'd'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-1]
            elif 'lpz' in lineSplit[i]: # numeric -> Z(log2(numeric+1))
                symbol = 'lpz'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-3]
            elif 'lp' in lineSplit[i]: # numeric -> log2(numeric+1)
                symbol = 'lp'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-2]
            elif 'lz' in lineSplit[i]: # numeric -> Z(log2(numeric))
                symbol = 'lz'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-2]
            elif 'l' in lineSplit[i]: # numeric -> log2(numeric)
                symbol = 'l'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-1]
            elif 'z' in lineSplit[i]: # numeric -> Z(numeric)
                symbol = 'z'
                lineSplit[i] = lineSplit[i][:len(lineSplit[i])-1]
            
            # check start and end number
            start = int(lineSplit[i].split('~')[0])
            end = int(lineSplit[i].split('~')[1])

            # add this range
            for j in range(start, end+1):
                result += ' ' + str(j) + symbol
    
        else: # not in the range form  (just append)  
            result += ' ' + lineSplit[i]

    return result

if __name__ == '__main__':
    f = open('input_output_info_original.txt', 'r')
    flines = f.readlines()
    f.close()

    # training input line
    TrainI = flines[0].split('\n')[0].split(' ')
    TrainI_converted = convertLine(TrainI)
    
    # training output line
    TrainO = flines[1].split('\n')[0].split(' ')
    TrainO_converted = convertLine(TrainO)

    # test input line
    TestI = flines[2].split('\n')[0].split(' ')
    TestI_converted = convertLine(TestI)

    # test output line
    TestO = flines[3].split('\n')[0].split(' ')
    TestO_converted = TestO[0]
    for i in range(1, len(TestO)):
        
        if len(TestO[i]) > 1: # in the range form
            count = int(TestO[i][1:])
            for j in range(count): TestO_converted += ' ' + TestO[i][0]

        else: # not in the range form (just append)
            TestO_converted += ' ' + Test0[i]

    # text to write
    toWrite = TrainI_converted + '\n' + TrainO_converted + '\n' + TestI_converted + '\n' + TestO_converted

    f = open('input_output_info_converted.txt', 'w')
    f.write(toWrite)
    f.close()
