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

        optiInfo = open('optiInfoForMap_' + str(problemNo) + '_ver1.txt', 'r')
        optiInformation = optiInfo.readlines()
        optiInfo.close()

        savePrint = ''

        print('testing...')

        # 2. 실험 및 테스트
        for y in range(size): # 각 y좌표에 대하여
            for x in range(size): # 각 x좌표에 대하여

                sumTestThroughput = 0.0 # test throughput의 합계
                sumCorrectMaxThroughput = 0.0 # 정답의 throughput의 합계 (training data를 생성할 때와 같은 방법으로 최대 throughput 확인)

                for i in range(numTest):

                    wdList_ = wdList[i]
                    opti = [y, x]
                    optiY_test = y
                    optiX_test = x

                    # 2-1. 테스트 결과 받아오기
                    (throughput, HAPtime) = WPCN_helper_REAL.getThroughput(wdList_, opti, size, problemNo)
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
                    printForThis = ('test throughput for map ' + str(i) + ' [Y=' + str(optiY_test) + ' X=' + str(optiX_test)
                                    + ' HT=' + str(HAPtime) + '] : ' + str(round(throughput, 6)) + ' / ' + str(round(correctMaxThroughput, 6)) + ' ('
                                    + str(round(100.0 * throughput / correctMaxThroughput, 6)) + ' %)')
                    print(printForThis)
                    savePrint += printForThis + '\n'
                    if i < 10: print('')
                
                # 3. 평가
                percentage = round(100.0 * sumTestThroughput / sumCorrectMaxThroughput, 6)
                print('size:' + str(size) + ' test:' + str(numTest) + ' loc(y,x):' + str(y) + ',' + str(x)
                        + ' problem(0=Sum,1=Common):' + str(problemNo))
                print('total test throughput: ' + str(round(sumTestThroughput, 6)))
                print('total max  throughput: ' + str(round(sumCorrectMaxThroughput, 6)))
                print('percentage           : ' + str(percentage) + ' %')

                RL.append([size, numTest, str(y) + ',' + str(x), problemNo, percentage])

        # 현재까지의 평가 결과 출력
        print('\n <<< ESTIMATION RESULT >>>')
        print('No.\tsize\ttest\tproblem\tpercentage')

        saveResult = '\n <<< ESTIMATION RESULT >>>\nNo.\tsize\ttest\tloc_yx\tproblem\tpercentage\n'
                    
        for i in range(len(RL)):
            toPrint = (str(i) + '\t' + str(RL[i][0]) + '\t' + str(RL[i][1]) + '\t' + str(RL[i][2]) + '\t'
                    + str(RL[i][3]) + '\t' + (' ' if round(RL[i][4], 4) < 10.0 else '') + str(round(RL[i][4], 4)))
            print(toPrint)
            saveResult += (toPrint + '\n')
        print('\n')

        # 평가 결과 저장
        fSave = open('DL_WPCN_result_' + str(problemNo) + '_fixed_ver200511.txt', 'w')
        fSave.write(saveResult)
        fSave.close()

        fSavePrint = open('DL_WPCN_result_' + str(problemNo) + '_fixed_ver200511_print.txt', 'w')
        fSavePrint.write(savePrint)
        fSavePrint.close()

        # 학습 및 테스트 종료
        break
