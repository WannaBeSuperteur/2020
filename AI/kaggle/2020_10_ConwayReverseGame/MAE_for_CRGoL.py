import numpy as np

# available for BINARY (0 or 1) values only
# fn            : file name
# thresholdList : list for thresholds (above->1, below->0)
# n             : number of values in each array pred = [...] or real = [...]
def readValidReport(fn, thresholdList, n):

    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # read each line and compute MAE
    MAE = [] # MAE values (for each threshold)
    for i in range(len(thresholdList)): MAE.append(0.0)
    lines = len(fl)-9 # number of lines
    
    for i in range(lines):
        if i % 50 == 0: print(i)
        
        # read predicted and real values
        pred = fl[i].split('pred = [')[1].split(']')[0] # pred = [...]
        real = fl[i].split('real = [')[1].split(']')[0] # real = [...]

        # convert to float
        predSplit = np.array(' '.join(pred.split()).split(' ')).astype('float')
        realSplit = np.array(' '.join(real.split()).split(' ')).astype('float')

        # compute MAE for each threshold
        for j in range(len(thresholdList)):
            for k in range(n):

                # pred is considered as 1 -> MAE=0 if real=1, MAE=1 if real=0
                if predSplit[k] > thresholdList[j]: MAE[j] += (1 - int(realSplit[k]))

                # pred is considered as 0 -> MAE=0 if real=0, MAE=1 if real=1
                else: MAE[j] += int(realSplit[k])

    # MAE <- MAE / (lines * n)
    for i in range(len(thresholdList)): MAE[i] = MAE[i] / (lines * n)
        
    # print MAE
    result = ''
    for i in range(len(thresholdList)):
        result_ = 'MAE when threshold = ' + str(round(thresholdList[i], 6)) + '\tis ' + str(round(MAE[i], 6))
        print(result_)
        result += result_ + '\n'

    # write to file
    f = open(fn[:len(fn)-4] + '_MAE.txt', 'w')
    f.write(result)
    f.close()

if __name__ == '__main__':
    thresholdList = []
    for i in range(1, 100): thresholdList.append(round(0.01*i, 6))
    readValidReport('valid_report_sub_0.txt', thresholdList, 400)
