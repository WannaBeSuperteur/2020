import numpy as np

if __name__ == '__main__':

    # read file
    fn = 'report_val.txt'
    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # extract values
    vals = []
    
    for i in range(len(fl)-9):
        pred = float(fl[i].split('[')[2].split(']')[0])
        real = float(fl[i].split('[')[3].split(']')[0])
        vals.append([pred, real])

    for i in range(100):
        threshold = 0.5 + 0.004 * i

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        pred_array = []
        real_array = []

        for j in range(len(vals)):
            if vals[j][0] >= threshold and vals[j][1] == 1:
                TP += 1
                pred_array.append(1)
                real_array.append(1)
            elif vals[j][0] < threshold and vals[j][1] == 0:
                TN += 1
                pred_array.append(0)
                real_array.append(0)
            elif vals[j][0] >= threshold and vals[j][1] == 0:
                FP += 1
                pred_array.append(1)
                real_array.append(0)
            elif vals[j][0] < threshold and vals[j][1] == 1:
                FN += 1
                pred_array.append(0)
                real_array.append(1)

        print(str(round(threshold, 4)) + '\t-> ' + str(TP) + ' / ' + str(TN) + ' / ' + str(FP) + ' / ' + str(FN) +
              ' accuracy=' + str(round(float((TP+TN)/(TP+TN+FP+FN)), 4)) +
              ' correl=' + str(round(np.corrcoef(pred_array, real_array)[0][1], 4)))
