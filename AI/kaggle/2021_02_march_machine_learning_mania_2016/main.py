import math
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_main as DL
import readData as RD
import random

if __name__ == '__main__':

    # read data
    regular_array = np.array(pd.read_csv('RegularSeasonCompactResults.csv'))
    tourney_array = np.array(pd.read_csv('TourneyCompactResults.csv'))

    all_array = np.concatenate((regular_array, tourney_array), axis=0)

    # number of teams
    N = 364
    N_start = 1101

    # extract raw_result : [Wteam, Lteam, 1, Wloc], [Lteam, Wteam, 0, Wloc]
    try:
        _ = open('raw_result.txt', 'r')
        _.close()

        _ = open('raw_result_test.txt', 'r')
        _.close()

    except:

        # TRAINING DATA : raw_result.txt
        
        raw_result = []

        # for [Wteam, Lteam, 1, Wloc]
        for i in range(len(all_array)):
            raw_result.append([all_array[i][2], all_array[i][4], 1, all_array[i][6]])

        # for [Lteam, Wteam, 0, Wloc]
        for i in range(len(all_array)):
            raw_result.append([all_array[i][4], all_array[i][2], 0, all_array[i][6]])

        RD.saveArray('raw_result.txt', raw_result, '\t', 500)

        # TEST DATA : raw_result_test.txt

        raw_result_test = []

        test_array = np.array(pd.read_csv('SampleSubmission.csv'))

        for i in range(len(test_array)):
            IDsplit = test_array[i][0].split('_')
            
            team0_id = int(IDsplit[1])
            team1_id = int(IDsplit[2])
            
            raw_result_test.append([team0_id, team1_id])

        RD.saveArray('raw_result_test.txt', raw_result_test, '\t', 500)

    # team_info : array for [teamID, total, win, lose, win_rate,
    #                        A_win, H_win, N_win, A_lose, H_lose, N_lose, A_win_rate, H_win_rate, N_win_rate]
    try:
        _ = open('team_info.txt', 'r')
        _.close()

    except:
        team_info = []
        for i in range(N):
            team_info.append([N_start + i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        for i in range(len(raw_result)):

            team0 = raw_result[i][0]
            team1 = raw_result[i][1]
            winLoc = raw_result[i][3]

            # team0 (win) vs. team1
            if raw_result[i][2] == 1:
                team_info[team0 - N_start][1] += 1 # update total
                team_info[team0 - N_start][2] += 1 # update win

                # location: team0 as win, team1 as lose
                if winLoc == 'A':
                    team_info[team0 - N_start][5] += 1
                    team_info[team1 - N_start][8] += 1
                elif winLoc == 'H':
                    team_info[team0 - N_start][6] += 1
                    team_info[team1 - N_start][9] += 1
                elif winLoc == 'N':
                    team_info[team0 - N_start][7] += 1
                    team_info[team1 - N_start][10] += 1

            # team0 (lose) vs. team1
            else:
                team_info[team0 - N_start][1] += 1 # update total
                team_info[team0 - N_start][3] += 1 # update lose

                # location: team0 as lose, team1 as win
                if winLoc == 'A':
                    team_info[team0 - N_start][8] += 1
                    team_info[team1 - N_start][5] += 1
                elif winLoc == 'H':
                    team_info[team0 - N_start][9] += 1
                    team_info[team1 - N_start][6] += 1
                elif winLoc == 'N':
                    team_info[team0 - N_start][10] += 1
                    team_info[team1 - N_start][7] += 1

        # update win_rate, A_win_rate, H_win_rate and N_win_rate
        for i in range(N):
            team_info[i][4] = team_info[i][2] / (team_info[i][2] + team_info[i][3])

            try:
                team_info[i][11] = team_info[i][5] / (team_info[i][5] + team_info[i][8])
            except:
                team_info[i][11] = 0.5
                
            try:
                team_info[i][12] = team_info[i][6] / (team_info[i][6] + team_info[i][9])
            except:
                team_info[i][12] = 0.5
                
            try:
                team_info[i][13] = team_info[i][7] / (team_info[i][7] + team_info[i][10])
            except:
                team_info[i][13] = 0.5

        RD.saveArray('team_info.txt', team_info, '\t', 500)

    # final : using raw_result and team_info
    # save as [team0_total, team0_win, team0_lose, team0_winrate,
    #          team0_A_win, team0_H_win, team0_N_win,
    #          team0_A_lose, team0_H_lose, team0_N_lose,
    #          team0_A_winRate, team0_H_winRate, team0_N_winRate,
    #          team1_total, ..., team1_N_winRate, result]
    try:
        _ = open('final_input.txt', 'r')
        _.close()

        _ = open('final_output.txt', 'r')
        _.close()

        _ = open('final_input_test.txt', 'r')
        _.close()

    except:

        # TRAINING INPUT : using raw_result
        
        final_input = []
        final_output = []
        
        for i in range(len(raw_result)):
            
            team0 = raw_result[i][0]
            team1 = raw_result[i][1]
            team0_ = team0 - N_start
            team1_ = team1 - N_start
            
            final_input.append([team_info[team0_][1], team_info[team0_][2], team_info[team0_][3],
                                team_info[team0_][4], team_info[team0_][5], team_info[team0_][6],
                                team_info[team0_][7], team_info[team0_][8], team_info[team0_][9],
                                team_info[team0_][10], team_info[team0_][11], team_info[team0_][12],
                                team_info[team0_][13],
                                team_info[team1_][1], team_info[team1_][2], team_info[team1_][3],
                                team_info[team1_][4], team_info[team1_][5], team_info[team1_][6],
                                team_info[team1_][7], team_info[team1_][8], team_info[team1_][9],
                                team_info[team1_][10], team_info[team1_][11], team_info[team1_][12],
                                team_info[team1_][13]])
            final_output.append([raw_result[i][2]])

        # TEST INPUT : using final_input_test

        final_input_test = []

        for i in range(len(raw_result_test)):

            team0 = raw_result_test[i][0]
            team1 = raw_result_test[i][1]
            team0_ = team0 - N_start
            team1_ = team1 - N_start
            
            final_input_test.append([team_info[team0_][1], team_info[team0_][2], team_info[team0_][3],
                                    team_info[team0_][4], team_info[team0_][5], team_info[team0_][6],
                                    team_info[team0_][7], team_info[team0_][8], team_info[team0_][9],
                                    team_info[team0_][10], team_info[team0_][11], team_info[team0_][12],
                                    team_info[team0_][13],
                                    team_info[team1_][1], team_info[team1_][2], team_info[team1_][3],
                                    team_info[team1_][4], team_info[team1_][5], team_info[team1_][6],
                                    team_info[team1_][7], team_info[team1_][8], team_info[team1_][9],
                                    team_info[team1_][10], team_info[team1_][11], team_info[team1_][12],
                                    team_info[team1_][13]])            

        # normalize each column of final_input

        # using average and stddevs of "final_input" ONLY
        avgs = []
        stddevs = []
        cols = 26

        for i in range(cols):
            thisCol = np.array(final_input)[:, i]
            avgs.append(np.mean(thisCol))
            stddevs.append(np.std(thisCol))

        # normalize
        for i in range(len(final_input)):
            for j in range(cols):
                final_input[i][j] = (final_input[i][j] - avgs[j]) / stddevs[j]

        for i in range(len(final_input_test)):
            for j in range(cols):
                final_input_test[i][j] = (final_input_test[i][j] - avgs[j]) / stddevs[j]

        RD.saveArray('final_input.txt', final_input, '\t', 500)
        RD.saveArray('final_output.txt', final_output, '\t', 500)
        RD.saveArray('final_input_test.txt', final_input_test, '\t', 500)

    # execute deep learning
    TRI = 'final_input.txt'
    TRO = 'final_output.txt'
    TEI = 'final_input_test.txt'
    TEO = 'final_output_test.txt'

    TE_real = None
    TE_report = None
    VAL_rate = 0.0
    VAL_report = 'report_val.txt'
    modelConfig = 'model_config.txt'

    # user data
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    epoch = int(input('epoch'))
    printed = int(input('printed? (0 -> do not print)'))

    DL.deepLearning(TRI, TRO, TEI, TEO,
                    TE_real, TE_report, VAL_rate, VAL_report, modelConfig,
                    deviceName, epoch, printed, 'model')
