# 목적:
# WPCN에서 HAP(=drone)의 위치와 충전 시간 할당 방법을 최적화한다.
# 논문: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7337464
#       Placement Optimization of Energy and Information Access Points in Wireless Powered Communication Networks
#       Suzhi Bi, Member, IEEE, and Rui Zhang, Senior Member, IEEE
#       Using Algorithm 5. Greedy algorithm for HAP placement optimization

# map.txt의 구성:
# (맵 세로길이) (맵 가로길이) (wireless device의 개수) (train 개수) (test 개수)

# convex/concave function이므로 위치가 바뀜에 따라 THROUGHPUT의 값이 convex/concave하게 바뀜

import math
import random
import sys
import WPCN_helper_REAL

## find L2 norm of array
def L2norm(array):
    result = 0
    for i in range(len(array)): result += pow(array[i], 2)
    return math.sqrt(result)

## add/subtract/mul/div array
# option: 0(add), 1(subtract), 2(mul), 3(div)
def arrayOp(array0, array1, option):
    assert(len(array0) == len(array1))

    result = []
    for i in range(len(array0)):
        if option == 0: result.append(array0[i] + array1[i])
        elif option == 1: result.append(array0[i] - array1[i])
        elif option == 2: result.append(array0[i] * array1[i])
        elif option == 3: result.append(array0[i] / array1[i])
    return result

# check if two arrays are equal
def arrayEqual(array0, array1):
    assert(len(array0) == len(array1))
    
    for i in range(len(array0)):
        if array0[i] != array1[i]: return False

    return True

# distance
def distance(array0, array1):
    assert(len(array0) == len(array1))
    
    result = 0
    for i in range(len(array0)): result += pow(array0[i]-array1[i], 2)
    return math.sqrt(result)

# Given j_k(b)'s, solve (20) for optimal AP placement vj*'s (=v1)
# max(t, V^N)(t) s.t. Lk - a1,k - a2,k*||v_(j_k) - w_k||^dU >= t, k=1,...,K, b^l <= v_j <= b^h, j=1,...,N
# that is, max(t, V^N)(t) s.t. Lk - a1,k - a2,k*||v_1 - w_k||^dU >= t, k=1,...,K, b^l <= v_1 <= b^h
# -> t should be minimum value of Lk - a1,k - a2,k*||v_1 - w_k||^dU
# -> So, the goal is to find "the V^N that maximizes (min value of Lk - a1,k - a2,k*||v_1 - w_k||^dU)"
# **** None for NOT FEASIBLE ****

# min value of Lk - a1,k - a2,k*||v_1k - w_k||^dU
def minOfT(L, a1, a2, w, dU, v1):
    K = len(L)
    
    result = L[0] - a1[0] - a2[0]*pow(L2norm(arrayOp(v1, w[0], 1)), dU)
    for k in range(1, K): result = min(result, L[k] - a1[k] - a2[k]*pow(L2norm(arrayOp(v1, w[k], 1)), dU))

    return result

# main function for find optimal v1 using Gradient Descent Method
def findMaxV(L, a1, a2, w, dU, v1):
    K = len(L)

    # find optimal v1 using Gradient Descent Method
    lr = 300000000.0
    for i in range(7000): # 7000 iterations
        minOfTgivenV1 = minOfT(L, a1, a2, w, dU, v1) # minOfT by this v1

        v1_0 = [v1[0]+0.000001, v1[1]] # add 1.0e-6 to 1st element
        v1_1 = [v1[0], v1[1]+0.000001] # add 1.0e-6 to 2nd element
        minOfTgivenV1_0 = minOfT(L, a1, a2, w, dU, v1_0) # minOfT by v1_0
        minOfTgivenV1_1 = minOfT(L, a1, a2, w, dU, v1_1) # minOfT by v1_1

        # if i < 100 or i % 50 == 0: print(i, v1, minOfTgivenV1)

        # update v
        v1[0] += lr * (minOfTgivenV1_0 - minOfTgivenV1)
        v1[1] += lr * (minOfTgivenV1_1 - minOfTgivenV1)

    return v1

## return a Feasible Solution of Problem (25) in the paper
# max(t, u)t s.t. (t + u - L) * ||u-wk||^dD <= p
# where k in {W1 U W2 U ... U Wi} = {W1} because there is only 1 HAP, u^l <= u_i(=u_1) <= u^h
# where u_i,k = min(u_(i-1),k, a1,k+a2,k*||ui-wk||^dU)
# ->    u_1,k = min(u_0,k, a1,k+a2,k*||u1-wk||^dU)

## using Algorithm 3. Trial-and-error method for N AP placement
# input: K WD locations and 1 EN locations
# output: locations of N APs [v1, ..., vN]
def feasibleSol(wdList_, HAPloc_, L, a1, a2, dU):
    K = len(wdList_)
    
    # Separate WDs into N cluster, and place each AP at a cluster center
    # use vj(0)'s to denote initial AP locations
    # -> no need to separate because the number of EN is 1
    
    v1 = [0, 0] # vj(0)'s (=v1) -> initialize as all 0's

    L = [] # values of Lk where k=1,...,K
    w = [] # values of wk where k=1,...,K (location of WD k)
    for k in range(K): w.append(wdList_[k])
    u = HAPloc_ # value of ui when i=1,2,...,M -> value of u1 (i=1) -> location of HAP

    # reference: VI. Simulation Results
    n = 0.51 # harvesting circuit efficiency (1.0 in report/200219_WPCN.pptx)
    Ad = [] # downlink antenna power gain for each WD k (hi = gi = 0.001 * 1.0^2 * d^(-2) = 0.001*d^(-2) in report/200219_WPCN.pptx)
    for k in range(K): Ad.append(3.0)
    fd = 915000000 # carrier frequency (915 MHz)
    P0 = 1.0 # transmit power (20.0 in report/200219_WPCN.pptx)
    dD = 2.2 # dD >= 2 denotes the path loss exponent in DL
    b = [] # b = [Ad*(3*10^8)/(4*PI*fd)]^dD for each WD k
    for k in range(K): b.append(pow(Ad[k]*(300000000/(4*3.141592654*fd)), dD))

    # With vj(0)'s, calculate j_k(b)'s using (6)
    jb = [] # j_k(b)'s for each k
    for k in range(K):
        jb.append(1) # j_k = argmin(j=1,...,N)||vj-wk||, k=1,...,K = argmin(j=1)||vj-wk|| = 1

    # With ui's, calculate Lk's using (5). (flag = 1)
    # Lk = n*b*P0*Sum(i=1,M)||ui-wk||^-dD, k=1,...,K
    #    = n*b*P0*||u1-wk||^-dD, k=1,...,K because i=1
    for k in range(K): L.append(n * b[k] * P0 * pow(L2norm(arrayOp(u, w[k], 1)), -dD))
    flag = 1

    while flag == 1:
        
        # Given j_k(b)'s, solve (20) for optimal AP placement vj*'s (=v1)
        # max(t, V^N)(t) s.t. Lk - a1,k - a2,k*||v_(j_k) - w_k||^dU >= t, k=1,...,K, b^l <= v_j <= b^h, j=1,...,N
        v1op = findMaxV(L, a1, a2, w, dU, v1)

        # Given vj*'s (=v1), calculate j_k(a)'s using (6)
        ja = [] # j_k(a)'s for each k
        for i in range(K):
            ja.append(1) # j_k = argmin(j=1,...,N)||vj-wk||, k=1,...,K = argmin(j=1)||v1-wk|| = 1

        # check if j_k(a) == j_k(b) for all k
        jabSame = [] # jabSame[i] = True if and only if j_a[i] == j_a[b], otherwise False
        for i in range(K):
            if arrayEqual(ja, jb): jabSame.append(True)
            else: jabSame.append(False)

        # if j_k(a) != j_k(b) for some k, update j_k(b) = j_k(a) for k=1,2,...,K
        localOptFound = True # local optimum is found if and only if j_k(a) == j_k(b) for all k
        for k in range(K):
            if jabSame[k] == False: # j_k(a) != j_k(b) for some k
                jb[k] = ja[k]
                localOptFound == False

        # a local optimum is found, then return optimal vj*'s = v1op
        if localOptFound == True:
            flag = 0
            return v1op

# 진행
if __name__ == '__main__':

    # 사용자 입력
    problemNo = int(input('0->sum throughtput maximization, 1->common throughtput maximization'))
    readOrWrite = int(input('0->read files, 1->create new files'))

    # 0. 파일을 읽어서 size 값과 train 개수, test 개수 구하기
    # 파일 형식 : (맵 세로길이) (맵 가로길이) (wireless device의 개수) (train 개수) (test 개수)
    # train 개수: 트레이닝할 데이터(맵)의 개수
    # test 개수 : 테스트할 데이터(맵)의 개수
    file = open('map.txt', 'r')
    read = file.readlines()
    readSplit = read[0].split(' ')
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    
    height = int(readSplit[0]) # 맵의 세로길이 (= 가로길이 = size)
    size = height
    numTrain = int(readSplit[3]) # 트레이닝할 데이터(맵)의 개수 (실제로는 트레이닝하지 않으며 파일 인덱싱에 사용됨)
    numTest = int(readSplit[4]) # 테스트할 데이터(맵)의 개수
    
    file.close()
    
    # 입력 정보
    originalScreen = [] # 각 맵의 스크린
    wdList = [] # 각 맵에서 wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]

    # 평가 결과 저장
    RL = []

    # 1. 테스트용 맵 생성
    # 맵의 구성: .(공간), H(HAP), W(wireless device)
    if readOrWrite == 1: # 맵 파일을 쓰고 그 맵으로 테스트
        print('generating maps...')
        for i in range(numTest):
            initScreen_ = WPCN_helper_REAL.initScreen(size, False)
            
            originalScreen.append(initScreen_[0])
            wdList.append(initScreen_[2])

            # 맵 파일 작성
            f = open('originalMaps/DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'w')
            for j in range(size):
                for k in range(size):
                    if originalScreen[i][j][k] == -1: f.write('W') # wireless device
                    elif originalScreen[i][j][k] == 0: f.write('.') # space
                    elif originalScreen[i][j][k] == 1: f.write('H') # HAP (not exist in the original map)
                f.write('\n')
            f.close()
            
    else: # 기존 맵 파일 읽어서 기존 맵으로 테스트
        print('reading maps...')
        for i in range(numTrain, numTrain+numTest):
            f = open('originalMaps/DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'r')
            map_ = f.readlines() # 맵을 나타낸 배열
            f.close()
            
            for j in range(len(map_)): map_[j] = map_[j].replace('\n', '') # 각 줄마다 개행문자 제거

            # originalScreen과 wdList 읽어오기
            wdListTemp = []
            thisScreen = [[0] * size for j in range(size)] # originalScreen에 추가할 맵(스크린)
            for j in range(size):
                for k in range(size):
                    if map_[j][k] == 'W':
                        thisScreen[j][k] = -1 # wireless device
                        wdListTemp.append([j, k]) # wdListTemp에 추가
                    elif map_[j][k] == '.': thisScreen[j][k] = 0 # space
                    elif map_[j][k] == 'H': thisScreen[j][k] = 1 # HAP (not exist in the original map)
                    
            originalScreen.append(thisScreen)
            wdList.append(wdListTemp)

    # 2~3. 논문에 나온 방법대로 실험 및 테스트
    while True:

        optiInfo = open('optiInfoForMap/optiInfoForMap_' + str(problemNo) + '_200519.txt', 'r')
        optiInformation = optiInfo.readlines()
        optiInfo.close()

        savePrint = ''
        saveResult = ''

        print('testing...')

        # 2. 실험 및 테스트
        for mode in range(1):

            sumTestThroughput = 0.0 # test throughput의 합계
            sumCorrectMaxThroughput = 0.0 # 정답의 throughput의 합계 (training data를 생성할 때와 같은 방법으로 최대 throughput 확인)

            if mode == 0:
                print(' < ORIGINAL >')
                savePrint += '< mode 0: original >\n'
                saveResult += '< mode 0: original >\n'
            elif mode == 1:
                print(' < ROUNDING >')
                savePrint += '\n< mode 1: rounding >\n'
                saveResult += '\n< mode 1: rounding >\n'
            
            for i in range(numTest):

                uStar = None # initialize solution u0*

                # 2-0. HAP의 최적의 x, y좌표 (optiX_test, optiY_test) 구하기

                # list of wireless device in i-th test data (in the form of [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]])
                wdList_ = wdList[i]
                if i < 10: print('wdList of map ' + str(i) + ' : ' + str(wdList_))

                # count of wireless device in i-th test data
                wdCount = len(wdList_)

                # do not have to cluster wireless devices because the number of HAP is 1
                # for i=1 to M do : do not use because M=1

                L = [] # value of L0,0, L0,1, ..., L0,(n-1)
                U = [] # value of U0,0, U0,1, ..., U0,(n-1)
                a1 = [] # value of a1,0, a1,1, ..., a1,(n-1) (constant circuit power of WD k)
                a2 = [] # value of a2,0, a2,1, ..., a2,(n-1) (parameter related to transmission strategy used in UL communication)
                dU = 2.5 # UL channel path loss exponent
                HAPloc_ = [0.5, 0.5] # location of HAP (initialized as (0.5, 0.5))

                # append a1,0 and a2,0 values for each WD - reference: VI. Simulation Results
                for k in range(len(wdList_)):
                    a1.append(0.00005)
                    a2.append(0.0000014)

                assumption = [] # 0 if (a) is TRUE, and 1 if (b) is TRUE

                # Using Algorithm 5. Greedy algorithm for HAP placement optimization
                # value of i is always 1 because there is only 1 HAP
                for j in range(len(wdList_)):
                    
                    # Update L(i-1),k and U(i-1),k using (19) and (24) where i=1 -> L0,k and U0,k
                    L.append(0)
                    U.append(2147483647) # infinite

                StopFlag = 0
                
                while True:
                    
                    # LB <- -K, UB -> K where K is sufficiently large
                    LB = -20
                    UB = 20

                    while True:
                        t = (UB + LB)/2 # ti <- (UB+LB)/2
                        
                        # Given ti, convert (25) into convex problem using procedures in Section V.B
                        # find feasible solutions of the convex problem: solution of (25)
                        sol = feasibleSol(wdList_, HAPloc_, L, a1, a2, dU) # using Algorithm 3: Trial-and-error method for N AP placement
                        
                        if sol != None: # Problem (25) is feasible given t:
                            LB = t
                            uStar = sol
                        else:
                            print(LB, UB)
                            UB = t

                        if abs(UB-LB) < 0.000001: break

                    # check if all K assumptions are valid
                    # Assume WD k satisfies condition (a) or (b)
                    countA = 0 # (a) u(i-1)k < a1,k + a2,k*||ui* - wk||^dU is TRUE
                    countB = 0 # (b) u(i-1)k >= a1,k + a2,k*||ui* - wk||^dU is TRUE

                    # check for assumptions for each wireless device
                    assumption = [] # initialize assumption array
                    
                    for k in range(len(wdList_)):
                        
                        # (a) u(i-1)k < a1,k + a2,k||ui* - wk||^dU
                        # (a) u0k < a1,k + a2,k||u0* - wk||^dU because there is only 1 HAP
                        if U[k] < a1[k] + a2[k]*pow(L2norm(arrayOp(uStar, wdList_[k], 1)), dU):
                            assumption.append(0)
                            countA += 1
                            
                        # (b) u(i-1)k >= a1,k + a2,k||ui* - wk||^dU
                        # (b) u0k >= a1,k + a2,k||u0* - wk||^dU because there is only 1 HAP
                        else:
                            assumption.append(1)
                            countB += 1 

                    # set StopFlag 1 when all K assumptions are valid
                    if countA == 0 or countB == 0: # all K assumptions are valid:
                        StopFlag = 1

                        [optiY_test, optiX_test] = [uStar[0], uStar[1]] # i-th HAP location = ui* (only 1 HAP -> u0*)

                    else:
                        StopFlag = 0
                        
                        # For each WD violating the assumption, switch the assumption from (a) to (b) or (b) to (a)
                        countA = len(wdList_)
                        countB = 0
                        for j in range(len(wdList_)): assumption[j] = 1 # switch (a) to (b)

                    if StopFlag == 1: break

                # 2-1. 테스트 결과 받아오기
                (throughput, HAPtime) = WPCN_helper_REAL.getThroughput(wdList_, uStar, size, problemNo)
                sumTestThroughput += throughput
                testScreen = originalScreen[i] # 테스트할 스크린

                # 2-2. testScreen과 optiScreen 중 일부 출력
                if i < 10: # 처음 10개의 map에 대해서만 출력
                    # testScreen 출력
                    print(' --- test input screen for map ' + str(i) + ' ---')
                    toPrint = ''
                    for j in range(size):
                        toPrint += '['
                        for k in range(size):
                            if testScreen[j][k] == -1: toPrint += 'W  '
                            else: toPrint += '.  '
                        toPrint += ']\n'
                    print(toPrint)

                # 2-3. 정답과 비교
                # 최적의 HAP의 위치 및 할당 시간 찾기
                # HAP의 위치 및 할당 시간에 따른 throughput의 최댓값
                correctMaxThroughput = float(optiInformation[(numTrain + i) * (size + 1)].split(' ')[5])
                sumCorrectMaxThroughput += correctMaxThroughput

                # 정답과 비교
                printForThis = ('test throughput for map ' + str(i) + ' [Y=' + str(round(optiY_test, 3)) + ' X=' + str(round(optiX_test, 3)) + ' HT=' + str(HAPtime) + '] : '
                                + str(round(throughput, 6)) + ' / ' + str(round(correctMaxThroughput, 6)) + ' ('
                                + str(round(100.0 * throughput / correctMaxThroughput, 6)) + ' %)')
                print(printForThis)
                savePrint += printForThis + '\n'
                if i < 10: print('')
            
            # 3. 평가
            percentage = round(100.0 * sumTestThroughput / sumCorrectMaxThroughput, 6)
            print('size:' + str(size) + ' test:' + str(numTest)
                    + ' problem(0=Sum,1=Common):' + str(problemNo))
            print('total test throughput: ' + str(round(sumTestThroughput, 6)))
            print('total max  throughput: ' + str(round(sumCorrectMaxThroughput, 6)))
            print('percentage           : ' + str(percentage) + ' %')

            RL.append([size, numTest, problemNo, percentage])

            # 현재까지의 평가 결과 출력
            print('\n <<< ESTIMATION RESULT >>>')
            print('No.\tsize\ttest\tproblem\tpercentage')

            saveResult += '\n <<< ESTIMATION RESULT >>>\nNo.\tsize\ttest\tproblem\tpercentage\n'
                        
            for i in range(len(RL)):
                toPrint = (str(i) + '\t' + str(RL[i][0]) + '\t' + str(RL[i][1]) + '\t'
                        + str(RL[i][2]) + '\t' + (' ' if round(RL[i][3], 4) < 10.0 else '') + str(round(RL[i][3], 4)))
                print(toPrint)
                saveResult += (toPrint + '\n')
            print('\n')

        # 평가 결과 저장
        fSave = open('DL_WPCN_result_' + str(problemNo) + '_paper_ver200519.txt', 'w')
        fSave.write(saveResult)
        fSave.close()

        fSavePrint = open('DL_WPCN_result_' + str(problemNo) + '_paper_ver200519_print.txt', 'w')
        fSavePrint.write(savePrint)
        fSavePrint.close()

        # 학습 및 테스트 종료
        break
