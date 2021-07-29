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
def getDistribArray(data, option, categoryNo):
    numSaleMonth = 80
    numSaleRepairDif = 60 # (repairMonth - saleMonth), at most 59 (less than 5 years)

    distrib_sale_dif = np.zeros((numSaleMonth, numSaleRepairDif))
    
    for i in range(len(data)):

        sale_month = (int(data[i][2].split('/')[0]) - 2003) * 12 + (int(data[i][2].split('/')[1]) - 3)
        sale_abs_month = int(data[i][2].split('/')[0]) * 12 + int(data[i][2].split('/')[1])
        repair_abs_month = int(data[i][3].split('/')[0]) * 12 + int(data[i][3].split('/')[1])

        dif_months = repair_abs_month - sale_abs_month
        
        if option == 'all':
            distrib_sale_dif[sale_month][dif_months] += 1
            
        elif option == 'M': # module category
            if categoryNo == int(data[i][0][1:]):
                distrib_sale_dif[sale_month][dif_months] += 1
                
        elif option == 'P': # component category
            if categoryNo == int(data[i][1][1:]):
                distrib_sale_dif[sale_month][dif_months] += 1
                
    return distrib_sale_dif

# fill with prediction for the cells whose (saleMonth) + (saleRepairDif) > 81 (Dec 2009) so no data
# where saleMonth     = 0 (Mar 2003) ... 79 (Oct 2009)
#       saleRepairDif = 0            ... 58

# count(saleMonth, saleRepairDif) : count of all the values with both (saleMonth) and (saleRepairDif)
# available                       : (saleMonth) + (saleRepairDif) <= 81

# thisSaleMonth                     = sum of [count(saleMonth, saleRepairDif) for all saleRepairDif available]
# thisSaleRepairDif                 = sum of [count(saleMonth, saleRepairDif) for all saleMonth     available]
# biggestAvailableMonth             = max(saleMonth     for all saleMonth     available)
# biggestAvailableSaleRepairDif     = max(saleRepairDif for all saleRepairDif available)
# sumAvailable                      = sum of [count(saleMonth, saleRepairDif)                     for
#                                             saleMonth     = 0 ... biggestAvailableMonth         and
#                                             saleRepairDif = 0 ... biggestAvailableSaleRepairDif]
# distrib[saleMonth][saleRepairDif] = thisSaleMonth * thisSaleRepairDif / sumAvailable
def fillDistribArray(distrib):

    numSaleMonth = len(np.array(distrib))
    numSaleRepairDif = len(np.array(distrib)[0])

    filled_distrib = np.zeros((numSaleMonth, numSaleRepairDif))

    for saleMonth in range(numSaleMonth):
        for saleRepairDif in range(numSaleRepairDif):

            # available (saleMonth + saleRepairDif <= 81)
            if saleMonth + saleRepairDif <= 81:
                filled_distrib[saleMonth][saleRepairDif] = distrib[saleMonth][saleRepairDif]

            # not available (saleMonth + saleRepairDif > 81)
            else:
                count = distrib[saleMonth][saleRepairDif]

                biggestAvailableMonth         = min(numSaleMonth     - 1, 81 - saleRepairDif)
                biggestAvailableSaleRepairDif = min(numSaleRepairDif - 1, 81 - saleMonth)

                thisSaleMonth     = np.sum(distrib[ saleMonth,                 :biggestAvailableSaleRepairDif + 1])
                thisSaleRepairDif = np.sum(distrib[:biggestAvailableMonth + 1,  saleRepairDif])
                sumAvailable      = np.sum(distrib[:biggestAvailableMonth + 1, :biggestAvailableSaleRepairDif + 1])

                try:
                    # invalid value test
                    test = thisSaleMonth * thisSaleRepairDif / sumAvailable
                    test = test + 1
                    if np.isnan(test) or np.isinf(test):
                        filled_distrib[saleMonth][saleRepairDif] = 0.0
                    else:
                        filled_distrib[saleMonth][saleRepairDif] = thisSaleMonth * thisSaleRepairDif / sumAvailable
                except:
                    filled_distrib[saleMonth][saleRepairDif] = 0.0

    return filled_distrib

# predict for the option and category No.
def predict(option, categoryNo):

    # make prediction
    distrib = pd.DataFrame(getDistribArray(repairTrain, option, categoryNo))
    filledDistrib = pd.DataFrame(fillDistribArray(np.array(distrib)))

    print(np.array(filledDistrib))

    numSaleMonth = len(np.array(distrib))
    numSaleRepairDif = len(np.array(distrib)[0])

    if option == 'all':
        catgNo = ''
    elif option == 'M':
        catgNo = str(categoryNo)
    elif option == 'P':
        catgNo = '%02d' % categoryNo
        
    distrib.to_csv('predict_' + option + catgNo + '.csv')
    filledDistrib.to_csv('predict_' + option + catgNo + '_filled.csv')

    # show the filled distribution array as image
    plt.axis([0, numSaleRepairDif, 0, numSaleMonth])
    plt.pcolor(filledDistrib)
    plt.title('prediction for ' + option + catgNo)

    plt.xlabel('dif of months (repair_month - sale_month)')
    plt.ylabel('sale month (Mar 2003~)')
    fileTitle = 'predict_' + option + catgNo + '_DS.png'
            
    plt.savefig(fileTitle)

if __name__ == '__main__':

    repairTrain = np.array(pd.read_csv('RepairTrain.csv'))

    # for entire
    predict('all', 0)

    # for M1, M2, ..., M9
    for i in range(9):
        predict('M', i+1)

    # for P01, P02, ..., P31
    for i in range(31):
        predict('P', i+1)
