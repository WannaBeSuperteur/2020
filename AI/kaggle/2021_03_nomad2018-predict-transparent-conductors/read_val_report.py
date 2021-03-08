import numpy as np
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# ref: https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

if __name__ == '__main__':

    RMSLE = 0
    
    for num in [0, 1]:

        # read file
        fn = 'report_val_' + str(num) + '.txt'
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

        print('RMSLE=' + str(round(rmsle(real, pred), 6)))
        RMSLE += rmsle(real, pred)

    print('average RMSLE=' + str(round(RMSLE / 2.0, 6)))
