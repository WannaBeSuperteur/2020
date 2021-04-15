import numpy as np
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
from sklearn.metrics import mean_squared_log_error

if __name__ == '__main__':

    # read file
    fns = ['report_val_funny.txt', 'report_val_useful.txt', 'report_val_cool.txt']

    # avg and std for funny / useful / cool
    avgs_and_stds = RD.loadArray('train_output_avg_and_std.txt')
    print(avgs_and_stds)

    for fn in range(len(fns)):
        f = open(fns[fn], 'r')
        fl = f.readlines()
        f.close()

        # extract values
        pred_array = []
        real_array = []
        
        for i in range(len(fl)-9):
            pred = float(fl[i].split('[')[2].split(']')[0]) * float(avgs_and_stds[1][fn]) + float(avgs_and_stds[0][fn])
            real = float(fl[i].split('[')[3].split(']')[0]) * float(avgs_and_stds[1][fn]) + float(avgs_and_stds[0][fn])

            # at least 0
            if pred < 0: pred = 0.0
            if real < 0: real = 0.0

            pred_array.append(pred)
            real_array.append(real)

        # convert to numpy array
        pred_array = np.array(pred_array)
        real_array = np.array(real_array)

        print('--------')
        print('file    :\n' + fns[fn])
        print('pred    :\n' + str(pred_array[:15]))
        print('real    :\n' + str(real_array[:15]))
        print('RMSE    =\n' + str(round(np.sqrt(np.mean((pred_array - real_array)**2)), 6)))
        print('RMSLE   =\n' + str(round(np.sqrt(mean_squared_log_error(real_array, pred_array)), 6)))
