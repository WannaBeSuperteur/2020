import matplotlib.pyplot as plt
import seaborn as seab
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_main as DL
import readData as RD

# get and show correlation
# input: pandas.DataFrame
def getCorr(dataFrame):
    df = dataFrame.corr()
    seab.clustermap(df,
                    annot=True,
                    cmap='RdYlBu_r',
                    vmin=-1, vmax=1)
    plt.show()

if __name__ == '__main__':
    team_info = RD.loadArray('team_info.txt', '\t')
    team_info = np.array(team_info)[:, 1:].astype(float)

    # normalize
    avgs = []
    stddevs = []
    cols = 14
    pca_cols = 8

    for i in range(cols):
        thisCol = np.array(team_info)[:, i]
        avgs.append(np.mean(thisCol))
        stddevs.append(np.std(thisCol))

    for i in range(len(team_info)):
        for j in range(cols):
            team_info[i][j] = (team_info[i][j] - avgs[j]) / stddevs[j]

    # convert into pandas.DataFrame
    team_info = pd.DataFrame(team_info)
    print(team_info)

    # print correlation between each column
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html

    #  0 = total
    #  1 = win
    #  2 = lose
    #  3 = win_rate
    #  4 = win_at_A
    #  5 = win_at_H
    #  6 = win_at_N
    #  7 = lose_at_A
    #  8 = lose_at_H
    #  9 = lose_at_N
    # 10 = winRate_at_A
    # 11 = winRate_at_H
    # 12 = winRate_at_N
    # 13 = total_seeds

    # get and show correlation
    getCorr(team_info)

    # PCA
    scaled = StandardScaler().fit_transform(team_info)
    pca = PCA(n_components=pca_cols)
    pca.fit(scaled)
    team_info_PCA = pca.transform(scaled)

    print(team_info_PCA)
    RD.saveArray('team_info_pca.txt', team_info_PCA, '\t', 500)

    # get and show correlation
    getCorr(pd.DataFrame(team_info_PCA))
