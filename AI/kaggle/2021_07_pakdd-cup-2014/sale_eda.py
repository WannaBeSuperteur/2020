import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# EDA from RepairTrain.csv
# option     : 'all', 'M' (module), 'P' (component)
# categoryNo : for example, 2 of 'M2', 6 of 'P06'

# y : count
# x : year/month (sale), Jan 2005 ~ Feb 2008, 38 months
def EDA(data, option, categoryNo):

    numSaleMonth = 38
    distrib_sale = np.zeros((numSaleMonth))

    for i in range(len(data)):

        sale_month = (int(data[i][2].split('/')[0]) - 2005) * 12 + (int(data[i][2].split('/')[1]) - 1)
        
        if option == 'all':
            distrib_sale[sale_month] += int(data[i][3])

        elif option == 'M': # module category
            if categoryNo == int(data[i][0][1:]):
                distrib_sale[sale_month] += int(data[i][3])

        # total monthly sales are ALL THE SAME for each component in the same month
        # (P01, Jun-05) -> 38,448, (P02, Jun-05) -> 38,448, ..., (P31, Jun-05) -> 38,448
        elif option == 'P': # component category
            if categoryNo == int(data[i][1][1:]):
                distrib_sale[sale_month] += int(data[i][3])

    # at least 0
    for i in range(numSaleMonth):
        distrib_sale[i] = max(0, distrib_sale[i])

    # show EDA result
    index = np.arange(numSaleMonth)
    plt.bar(index, distrib_sale)
    plt.title('sale EDA for ' + option + str(categoryNo))

    plt.xlabel('sale month (Jan 2005~)')
    plt.ylabel('sale count')
    fileTitle = 'EDA_sale_' + option + str(categoryNo) + '.png'
    
    plt.savefig(fileTitle)
    plt.clf()

if __name__ == '__main__':

    repairTrain = np.array(pd.read_csv('SaleTrain.csv'))

    EDA(repairTrain, 'all', 0)
    
    for i in range(9):
        EDA(repairTrain, 'M', i+1)
    for i in range(1):
        EDA(repairTrain, 'P', i+1)
