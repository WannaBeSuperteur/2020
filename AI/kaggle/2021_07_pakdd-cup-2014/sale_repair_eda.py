import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# EDA from RepairTrain.csv
# option     : 'all', 'M' (module), 'P' (component)
# categoryNo : for example, 2 of 'M2', 6 of 'P06'

# y : year/month (sale)  , Mar 2003 ~ Oct 2009, 80 months
# x : year/month (repair), Feb 2005 ~ Dec 2009, 59 months
def EDA(data, option, categoryNo):
    numSaleMonth = 80
    numRepairMonth = 59
    numSaleRepairDif = 60 # (repairMonth - saleMonth), at most 59 (less than 5 years)

    distrib_sale_repair = np.zeros((numSaleMonth, numRepairMonth))
    distrib_sale_dif = np.zeros((numSaleMonth, numSaleRepairDif))
    distrib_repair_dif = np.zeros((numRepairMonth, numSaleRepairDif))

    for i in range(len(data)):

        sale_month = (int(data[i][2].split('/')[0]) - 2003) * 12 + (int(data[i][2].split('/')[1]) - 3)
        repair_month = (int(data[i][3].split('/')[0]) - 2005) * 12 + (int(data[i][3].split('/')[1]) - 2)

        sale_abs_month = int(data[i][2].split('/')[0]) * 12 + int(data[i][2].split('/')[1])
        repair_abs_month = int(data[i][3].split('/')[0]) * 12 + int(data[i][3].split('/')[1])
        dif_months = repair_abs_month - sale_abs_month
        
        if option == 'all':
            distrib_sale_repair[sale_month][repair_month] += 1
            distrib_sale_dif[sale_month][dif_months] += 1
            distrib_repair_dif[repair_month][dif_months] += 1

        elif option == 'M': # module category
            if categoryNo == int(data[i][0][1:]):
                distrib_sale_repair[sale_month][repair_month] += 1
                distrib_sale_dif[sale_month][dif_months] += 1
                distrib_repair_dif[repair_month][dif_months] += 1

        elif option == 'P': # component category
            if categoryNo == int(data[i][1][1:]):
                distrib_sale_repair[sale_month][repair_month] += 1
                distrib_sale_dif[sale_month][dif_months] += 1
                distrib_repair_dif[repair_month][dif_months] += 1

    arrays = [distrib_sale_repair, distrib_sale_dif, distrib_repair_dif]
    titles = ['sale - repair', 'sale - dif', 'repair - dif']

    for i in range(3):
        distrib_df = pd.DataFrame(arrays[i])

        # show EDA result
        plt.axis([0, len(arrays[i][0]), 0, len(arrays[i])])
        plt.pcolor(distrib_df)
        plt.title(titles[i] + ' for ' + option + str(categoryNo))

        if i == 0:
            plt.xlabel('repair month (Feb 2005~)')
            plt.ylabel('sale month (Mar 2003~)')
            fileTitle = 'EDA_' + option + str(categoryNo) + '_RS.png'
        elif i == 1:
            plt.xlabel('dif of months')
            plt.ylabel('sale month (Mar 2003~)')
            fileTitle = 'EDA_' + option + str(categoryNo) + '_DS.png'
        elif i == 2:
            plt.xlabel('dif of months')
            plt.ylabel('repair month (Feb 2005~)')
            fileTitle = 'EDA_' + option + str(categoryNo) + '_DR.png'
            
        plt.savefig(fileTitle)

if __name__ == '__main__':

    repairTrain = np.array(pd.read_csv('RepairTrain.csv'))

    EDA(repairTrain, 'all', 0)
    
    for i in range(9):
        EDA(repairTrain, 'M', i+1)
    for i in range(31):
        EDA(repairTrain, 'P', i+1)
