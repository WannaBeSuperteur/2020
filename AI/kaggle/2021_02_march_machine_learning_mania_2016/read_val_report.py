import numpy as np

# ref: https://stackoverflow.com/questions/58511404/how-to-compute-binary-log-loss-per-sample-of-scikit-learn-ml-model
def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)

    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)

if __name__ == '__main__':

    # read file
    fn = 'report_val_pca.txt'
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

    # compute log loss
    logLoss = [logloss(x, y) for (x, y) in zip(real_array, pred_array)]
        
    print('correl=' + str(round(np.corrcoef(pred_array, real_array)[0][1], 4)) +
          ', log_loss=' + str(round(np.mean(logLoss), 6)))
