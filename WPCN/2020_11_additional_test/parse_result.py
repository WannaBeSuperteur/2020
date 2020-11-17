import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab

import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD

# corrCols: list of 2 columns that you want to know the correlation about
#           for example, [['X', 'Y'], ['thrput', 'maxThrput']]
def main(size, WDs, corrCols):
    print('size=' + str(size) + ' WDs=' + str(WDs))

    fn = 'DL_WPCN_forpaper_1_' + str(size) + '_' + str(WDs) + '_detail.txt'

    # read file
    f = open(fn, 'r')
    fl = f.readlines()
    f.close()

    # result array
    result = []

    # split data
    for i in range(len(fl)):
        thisLine = fl[i].split('\n')[0]

        # extract each column
        mapNo = int(thisLine.split(' ')[4])
        Y = float(thisLine.split('Y=')[1].split(' ')[0])
        X = float(thisLine.split('X=')[1].split(' ')[0])
        HT = float(thisLine.split('HT=')[1].split(']')[0])
        thrput = float(thisLine.split(' ')[9])
        maxThrput = float(thisLine.split(' ')[11])
        rate = float(thisLine.split('(')[1].split(' ')[0])

        result.append([mapNo, Y, X, HT, thrput, maxThrput, rate])

    # convert to dataFrame
    cols = ['mapNo', 'Y', 'X', 'HT', 'thrput', 'maxThrput', 'rate']
    df = pd.DataFrame(np.array(result), columns=cols)

    # print dataFrame
    print(df)

    # save data as file
    RD.saveArray(fn[:len(fn)-4] + '_data.txt', result)

    # display correlation
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    dfCorr = df.corr() # get correlation
    seab.clustermap(dfCorr,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

    # return correlations
    corrs = []
    for i in range(len(corrCols)):
        corrCol = corrCols[i]
        corr = dfCorr[corrCol[0]][corrCol[1]] # correlation

        n = 100 # number of samples = number of test data
        Z = np.arctanh(corr) # tanh^-1(corr)
        EL = Z - 1.96 * math.sqrt(1/(n-3)) # E(Z)_L = Z - 1.96 * sqrt(1/(n-3))
        EU = Z + 1.96 * math.sqrt(1/(n-3)) # E(Z)_U = Z + 1.96 * sqrt(1/(n-3))
        pL = math.tanh(EL) # p_L = tanh(E(Z)_L)
        pU = math.tanh(EU) # p_U = tanh(E(Z)_U)
        
        corrs.append([size, WDs, corrCol[0], corrCol[1], corr, pL, pU])

    return corrs

if __name__ == '__main__':
    corrs = []
    cols = ['mapNo', 'Y', 'X', 'HT', 'thrput', 'maxThrput', 'rate']

    corrCols = []
    for i in range(1, len(cols)):
        for j in range(i):
            corrCols.append([cols[i], cols[j]])
    
    for size in [8, 12, 16]:
        for WDs in [6, 10]:
            corrsTemp = main(size, WDs, corrCols)
            for i in range(len(corrsTemp)): corrs.append(corrsTemp[i])
                
    # save corrs
    RD.saveArray('corrs.txt', corrs)
