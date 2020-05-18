import math
import random
import numpy as np
from operator import itemgetter

# 스크린 표시 함수
# screen   : 스크린 (odd*odd size)
# score    : 현재 점수
# point    : HAP의 좌표
# wdList   : wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]
def displayScreen(screen, score, count, HAPtime):
    print('< 카운트: ' + str(count) + ' 점수: ' + str(round(score, 6)) + ' >')

    x = 1 - HAPtime # HAPtime = 1-x
    device = [x*HAPtime, x*x*HAPtime, x*x*x*HAPtime]
    print('< HAP의 충전 시간: ' + str(round(HAPtime, 3)) + ', WD의 충전 시간: '
          + str(round(device[0], 3)) + ', ' + str(round(device[1], 3)) + ', ' + str(round(device[2], 3)) + ', ... >')
    
    print(' +=' + '==='*len(screen[0]) + '=+')
    
    for i in range(len(screen)):
        temp = ' | '
        for j in range(len(screen[0])):
            if screen[i][j] == 1: temp += 'HAP' # HAP
            elif screen[i][j] == 0: temp += ' . ' # 공간
            elif screen[i][j] == -1: temp += 'W.D' # wireless device
        print(temp + ' | ')
    print(' +=' + '==='*len(screen[0]) + '=+')
    print('')

# 스크린 초기화
# 파일 이름 : map.txt
# 파일 형식 : (맵 세로길이) (맵 가로길이) (wireless device의 개수)
# size      : screen의 크기 (= 가로길이 = 세로길이)
# isHAP     : HAP가 맵에 존재하는가?
def initScreen(size, isHAP):
    originalScreen = [[0] * size for j in range(size)]
    file = open('map.txt', 'r')
    read = file.readlines()
    readSplit = read[0].split(' ')

    height = int(readSplit[0]) # 맵의 세로길이
    width = int(readSplit[1]) # 맵의 가로길이
    numWD = int(readSplit[2]) # wireless device의 개수
          
    file.close()

    assert(size == height and size == width)

    # HAP 배치하기
    if isHAP:
        HAPy = random.randint(0, size-1)
        HAPx = random.randint(0, size-1)
        originalScreen[HAPy][HAPx] = 1
        point = [HAPy, HAPx]
    else:
        point = None

    # wireless device 배치하기
    wdList = []
    for i in range(numWD):
        while True:
            WDy = random.randint(0, size-1)
            WDx = random.randint(0, size-1)
        
            if originalScreen[WDy][WDx] == 0:
                originalScreen[WDy][WDx] = -1
                wdList.append([WDy, WDx])
                break

    # originalScreen: 초기 스크린
    # point         : HAP의 좌표 
    # wdList        : wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]
    return [originalScreen, point, wdList]

# copy array
def copyArray(array):
    result = [0]*len(array)
    for i in range(len(array)): result[i] = array[i]
    return result

# print rounded array
def printRounded(array, n):
    result = [0]*len(array)
    for i in range(len(array)): result[i] = round(array[i], n)
    return str(result)

# throughput의 합(sum-throughput maximization) 또는 최솟값(common throughput maximization)을 구하는 함수
# (Gradient Descent Method를 이용하여 chargeTimeList를 최적화한 결과에서의 throughput의 합 또는 최솟값)

# wdList    : wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]
# point     : HAP의 좌표
# screenSize: screen의 크기 (= 가로길이 = 세로길이)
# problemNo : 0이면 sum throughtput maximization, 1이면 common throughtput maximization
def getThroughput(wdList, point, screenSize, problemNo):

    if problemNo == 0: result = 0.0 # Sum-throughput maximization (throughput의 합)
    elif problemNo == 1: result = 1000.0 # Common throughput maximization problem (throughput의 최솟값)
    
    pointY = point[0] # HAP의 세로좌표
    pointX = point[1] # HAP의 가로좌표

    mapSizeSquare = screenSize * screenSize
    lr = 10000000000.0 # learning rate

    # Find optimal time allocation for sum/common-throughput maximization problem using Gradient Descent method
    # for both HAP and each wireless device (HAP for index 0 and each device for rest)
    chargeTimeList = [1.0]*(1+len(wdList)) # time allocation list for HAP and each wireless device (sum of time = 1)
    # print('\n\n' + str(wdList) + '\n')
    for i in range(1000):

        thrputChange = [] # wdList의 각 값을 1만큼 증가시켰을 때 throughput의 변화량
        (originalThrput, None_0) = getThroughput_(wdList, point, chargeTimeList, problemNo) # 원래 Throughput
        # print(printRounded(chargeTimeList, 6) + ' ' + printRounded([originalThrput], 6))

        # wdList의 각 값을 1.0e-9만큼 증가시켜서 throughput이 얼마나 증가/감소하는지 확인하고,
        # 그 값만큼 chargeTimeList의 해당 부분을 증감시킨다.
        for j in range(1+len(wdList)):
            chargeTimeListCopy = copyArray(chargeTimeList) # 배열 복사
            chargeTimeListCopy[j] += 0.000000001
            (thrput, None_1) = getThroughput_(wdList, point, chargeTimeListCopy, problemNo) # wdList의 각 값을 변화시킨 후에 구한 Throughput

            # throughput의 변화량을 추가: log2(throughput / old throught)을 leaky relu처럼 적용
            thrputChange.append(max(0.01*math.log(thrput/originalThrput, 2.0), math.log(thrput/originalThrput, 2.0)))

        # chargeTimeList 갱신
        for j in range(1+len(wdList)):
            chargeTimeList[j] *= (pow(2.0, lr * thrputChange[j]))

        # print(i, thrput)

    # 충분히 많은 iteration 후의 chargeTimeList에 의한 결과값
    (finalThrput, HAPtime) = getThroughput_(wdList, point, chargeTimeList, problemNo)
    return (finalThrput, HAPtime)

# 각 wireless device에 대한 throughput 계산 (특정 chargeTimeList에서의 합 또는 최솟값)
def getThroughput_(wdList, point, chargeTimeList, problemNo):
    pointY = point[0] # HAP의 세로좌표
    pointX = point[1] # HAP의 가로좌표
    
    sumOfChargeTime = sum(chargeTimeList)

    if problemNo == 0: result = 0.0 # Sum-throughput maximization (throughput의 합)
    elif problemNo == 1: result = 1000.0 # Common throughput maximization problem (throughput의 최솟값)

    HAPtime = chargeTimeList[0] / sumOfChargeTime # time allocated to HAP
    
    # 각 wireless device에 대한 directReward와 그 합 구하기
    for i in range(len(wdList)):
        wdY = wdList[i][0] # wireless device의 가로 좌표
        wdX = wdList[i][1] # wireless device의 세로 좌표
        wdDistance = (wdY-pointY)*(wdY-pointY) + (wdX-pointX)*(wdX-pointX) # wireless device와 point와의 거리의 제곱

        # 해당 wireless device의 throughput
        # throughput: Ti * log2(1 + ri * T0/Ti)
        #           = Ti * log2(1 + S*hi*gi*Pa/(SNR*o^2) * T0/Ti)
        #           = Ti * log2(1 + S*(0.001*pi^2*(distance)^(-2))^2*Pa/(SNR*o^2) * T0/Ti)
        #           = Ti * log2(1 + S*0.000001*pi^4*(distance)^(-4)*Pa/(SNR*o^2) * T0/Ti)
        #           = Ti * log2(1 + 1.0*0.000001*pi^4*wdDistance^(-2)*20.0/(9.8*0.000001) * T0/Ti)
        #           = Ti * log2(1 + 100*pi^4/(49*wdDistance^2) * T0/Ti)
        
        # ri  : value of S*hi*gi*Pa/(SNR*o^2)
        # Ti  : time portion in each block alloated to user Ui (=chargeTime)
        # T0  : time portion in each block alloated to HAP (HAPTime)
        # S   : energy harvesting efficiency (S1=S2=...=S -> =1.0)
        # hi  : Channel power gain hi = |~hi|^2 where ~hi is a complex random variable
        #       modeling: hi = 0.001 * pi^2 * (distance)^-ad where ad=2.0
        # gi  : Channel power gain gi = |~gi|^2 where ~gi is a complex random variable
        #       modeling: gi = 0.001 * pi^2 * (distance)^-au where au=2.0
        # pi  : pi^2 is an exponentially distributed random varaible with unit mean
        #       -> pi = f(x) with x from 0 to 1, f(x) = sqrt(2^x * ln2), where distribution of x is uniform
        #       FOR EASY LEARNING, ASSUME THAT VALUE OF PI IS 1.0
        # Pa  : (large such that energy harvested due to receiver noise is negligible -> =20.0)
        # SNR : signal-to-noise (SNR) ratio (=9.8)
        # o   : value of o where z(A,i), received noise at HAP during slot i, follows CN(0, o^2) (=0.001)

        # max(wdDistance, 1): wdDistance must be at least 1.0
        pi = 1.0
        chargeTime = chargeTimeList[1+i] / sumOfChargeTime # time allocated to each user
        throughput = chargeTime * math.log(1 + (100*pow(pi, 4))/(49*pow(max(wdDistance, 1), 2)) * HAPtime/chargeTime, 2)
        
        if problemNo == 0: # Sum-throughput maximization (throughput의 합)
            result += throughput
        elif problemNo == 1: # Common throughput maximization problem (throughput의 최솟값)
            if result > throughput: result = throughput

        # print(throughput, chargeTime, mapSizeSquare, wdDistance)

    return (result, HAPtime)

# 소수점 이하 n자리까지 배열 출력, useRate이면 최댓값을 1로 함
def printArray(array, n, useRate):

    # useRate인 경우 최댓값을 1로 하여 상대값을 구함
    if useRate:
        maxValue = 0.0
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] > maxValue: maxValue = array[i][j]
        print('use relative value where max value(' + str(maxValue) + ') = 1.0')

    # 소수점 이하 n자리까지 배열 출력
    for i in range(len(array)):
        result = ''
        for j in range(len(array[0])):
            if useRate: # 최댓값에 대한 상대값
                array[i][j] = round(array[i][j] / maxValue, n)
            else: # 원래 값
                array[i][j] = round(array[i][j], n)

            result += str(array[i][j]) + ' '*(n+4-len(str(array[i][j])))
        print(result)

if __name__ == '__main__':
    maplist = input('input No. of maps (ex: 0 1 2 3 4 100 101 102 103 104)').split(' ')
    problemNo = int(input('input problem No. (0 or 1)'))

    file = open('map.txt', 'r')
    read = file.readlines()
    readSplit = read[0].split(' ')
    screenSize = int(readSplit[0]) # 맵의 길이 (가로=세로)
    file.close()

    # 각 map에 대하여
    for mapIndex in range(len(maplist)):
        i = int(maplist[mapIndex])

        # 각 지점의 throughput을 저장할 배열
        throughputList = [[0.0] * screenSize for j in range(screenSize)]

        ## 0. wdList 구하기
        wdList = []

        # wdList를 구하기 위한 맵 불러오기
        file_ = open('DL_WPCN_' + ('0' if i < 1000 else '') + ('0' if i < 100 else '') + ('0' if i < 10 else '') + str(i) + '.txt', 'r')
        map_ = file_.readlines() # 맵을 나타낸 배열
        file_.close()
        for j in range(len(map_)): map_[j] = map_[j].replace('\n', '') # 각 줄마다 개행문자 제거

        # wdList에 추가하기
        for j in range(screenSize):
            for k in range(screenSize):
                if map_[j][k] == 'W': wdList.append([j, k])

        ## 1. 각 지점에 대한 throughput 계산
        for j in range(screenSize):
            for k in range(screenSize):
                (thrput, noUse) = getThroughput(wdList, [j, k], screenSize, problemNo)
                throughputList[j][k] = thrput

        ## 2. 출력
        print('throughput list for map ' + str(i))
        if problemNo == 1: useRate = True # Common Throughput Maximization에서는 throughput이 매우 작으므로 상대값 사용
        else: useRate = False
        printArray(throughputList, 4, useRate)
        print('')
