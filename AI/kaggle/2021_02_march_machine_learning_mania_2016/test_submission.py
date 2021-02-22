import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

if __name__ == '__main__':

    regular_array = np.array(pd.read_csv('RegularSeasonCompactResults.csv'))
    tourney_array = np.array(pd.read_csv('TourneyCompactResults.csv'))

    all_array = np.concatenate((regular_array, tourney_array), axis=0)

    # N = (number of teams) = (1101 ~ 1364) = 364
    # make (N, N) array
    N = 364
    goals_array = np.zeros((N, N, 2))
    
    for i in range(len(all_array)):
        if i % 10000 == 0: print(i)
        
        Wteam = int(all_array[i][2])
        Wscore = int(all_array[i][3])
        Lteam = int(all_array[i][4])
        Lscore = int(all_array[i][5])

        # team[i][j] = [i score, j score] where i < j
        
        # Wteam < Lteam --> team[Wteam][Lteam] += [Wscore, Lscore]
        if Wteam < Lteam:
            goals_array[Wteam - 1101][Lteam - 1101][0] += Wscore
            goals_array[Wteam - 1101][Lteam - 1101][1] += Lscore

        # Wteam > Lteam -> team[Lteam][Wteam] += [Lscore, Wscore]
        else:
            goals_array[Lteam - 1101][Wteam - 1101][0] += Lscore
            goals_array[Lteam - 1101][Wteam - 1101][1] += Wscore

    # goals rate array : (i goals) / (i goals + j goals)
    goals_rate_array = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            
            if sum(goals_array[i][j]) == 0:
                goals_rate_array[i][j] = 0.5
            else:
                goals_rate_array[i][j] = goals_array[i][j][0] / sum(goals_array[i][j])

            # transformation: 1 - 2*(1 - x)^2 if x >= 0.5
            #                 2x^2 if x < 0.5
            if goals_rate_array[i][j] >= 0.5:
                goals_rate_array[i][j] = 1 - 2 * (1 - goals_rate_array[i][j])**2
            else:
                goals_rate_array[i][j] = 2 * (goals_rate_array[i][j])**2

    # prediction for 2016 season
    sample_submission = np.array(pd.read_csv('SampleSubmission.csv'))
    final_result = []

    for i in range(len(sample_submission)):
        info = sample_submission[i][0].split('_')

        teamA = int(info[1])
        teamB = int(info[2])

        final_result.append([goals_rate_array[teamA - 1101][teamB - 1101]])

    # write as file
    RD.saveArray('finalResult.txt', final_result, '\t', 500)
