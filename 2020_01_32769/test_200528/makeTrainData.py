fn = input('file name of raw data')
writefn = input('file name of train or test data')

f = open(fn, 'r')
fr = f.readlines()
f.close()

atLeast = int(input('value must be at least '))
colsEachTrain = int(input('number of columns of each training data'))
after = int(input('train and test outputs are N days later of last input data'))

toWriteToPaperTrain = '' # data to write at fn
toPrint = ''

for i in range(len(fr)):
    frs = fr[i].split('\t') # split each row by tab
    for j in range(len(frs)): frs[j] = int(frs[j]) # integer

    # using (colsEachTrain) consecutive data whose values are all >=(atLeast)
    for j in range(len(frs)-colsEachTrain+1-after):
        
        if frs[j] >= atLeast and frs[j+colsEachTrain-1+after] >= atLeast: # if values are all >=(atLeast)
            toWrite = '' # each training/test data

            # input data: (data[first+i]/data[(first+last)/2] - 1)
            for k in range(colsEachTrain):
                toWrite += str(frs[j+k] / frs[j+int(colsEachTrain/2)] - 1) + '\t'
                toPrint += str(frs[j+k]) + '\t'

            # output data: (data[last+7]/data[last] - 1)
            toWrite += str(frs[j+colsEachTrain-1+after] / frs[j+colsEachTrain-1] - 1) + '\n'
            toPrint += str(frs[j+colsEachTrain-1+after]) + '\n'

            toWriteToPaperTrain += toWrite

# write info
f = open('info.txt', 'w')
f.write('value must be at least ' + str(atLeast) +
        '\nnumber of columns of each training data is ' + str(colsEachTrain) +
        '\ntrain and test outputs are ' + str(after) + ' days later of last input data')
f.close()

# write to writefn
f = open(writefn, 'w')
f.write(toWriteToPaperTrain[:len(toWriteToPaperTrain)-len('\n')])
f.close()

# write to 'paper_toWrite.txt'
f = open('print_' + writefn, 'w')
f.write(toPrint[:len(toPrint)-len('\n')])
f.close()
            
