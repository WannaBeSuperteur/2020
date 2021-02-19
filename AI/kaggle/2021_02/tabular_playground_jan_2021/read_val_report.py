import numpy as np

if __name__ == '__main__':

    # read file
    fn = 'report_val.txt'
    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # extract values
    pred_array = []
    real_array = []
    
    for i in range(len(fl)-9):
        pred = float(fl[i].split('[')[2].split(']')[0])
        real = float(fl[i].split('[')[3].split(']')[0])

        pred_array.append(pred)
        real_array.append(real)

    print('RMSE=' + str(round(np.sqrt(np.mean((np.array(pred_array) - np.array(real_array))**2)), 6)))
