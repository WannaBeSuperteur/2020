import math
import random
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

# throughput의 합(sum-throughput maximization) 또는 최솟값(common throughput maximization)을 구하는 함수
# wdList    : wireless device의 위치를 저장한 배열, [[wd0Y, wd0X], [wd1Y, wd1X], ..., [wd(n-1)Y, wd(n-1)X]]
# point     : HAP의 좌표
# screenSize: screen의 크기 (= 가로길이 = 세로길이)
# HAPtime   : 전체 충전 시간(=1)에 대한 HAP의 충전 시간의 비율(=1-x)
# problemNo : 0이면 sum throughtput maximization, 1이면 common throughtput maximization
def getThroughput(wdList, point, screenSize, HAPtime, problemNo):

    if problemNo == 0: result = 0.0 # Sum-throughput maximization (throughput의 합)
    elif problemNo == 1: result = 1000.0 # Common throughput maximization problem (throughput의 최솟값)
    
    pointY = point[0] # HAP의 세로좌표
    pointX = point[1] # HAP의 가로좌표

    mapSizeSquare = screenSize * screenSize

    # 정렬을 위해 wdList의 각 원소에 3번째 원소(point와의 거리의 제곱)를 추가
    for i in range(len(wdList)):
        wdY = wdList[i][0] # wireless device의 가로 좌표
        wdX = wdList[i][1] # wireless device의 세로 좌표
        wdList[i] = [wdY, wdX, ((wdX-pointX)*(wdX-pointX) + (wdY-pointY)*(wdY-pointY))]

    # Sum-throughput maximization problem
    # 결과값은 각 WD의 throughput의 합으로 계산한다.
    if problemNo == 0:

        # 가까울수록 시간을 많이 할당해야 sum-throughput이 최대화되므로 충전시간은 x^(k+1) * (1-x)
        # k는 wdList를 가까운 순으로 정렬한 배열에서의 해당 wireless device의 인덱스

        # 정렬 (distance의 제곱 순)
        wdList = sorted(wdList, key=itemgetter(2), reverse=False)

    # Common throughput maximization problem
    # 결과값은 common throughput, 즉 각 WD의 throughput의 최솟값과 같다.
    elif problemNo == 1:

        # 멀수록 시간을 많이 할당해야 common throughput이 최대화되므로 충전시간은 x^(k+1) * (1-x)
        # k는 wdList를 먼 순으로 정렬한 배열에서의 해당 wireless device의 인덱스

        # 정렬 (distance의 제곱에 대한 역순)
        wdList = sorted(wdList, key=itemgetter(2), reverse=True)

    # 정렬된 배열을 이용하여 각 wireless device에 대한 directReward와 그 합 구하기
    for i in range(len(wdList)):
        wdY = wdList[i][0] # wireless device의 가로 좌표
        wdX = wdList[i][1] # wireless device의 세로 좌표
        wdDistance = wdList[i][2] # wireless device와 point와의 거리의 제곱

        x = 1.0 - HAPtime
        chargeTime = pow(x, i+1) * (1-x) # 충전 시간

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
        throughput = chargeTime * math.log(1 + (100*pow(pi, 4))/(49*pow(max(wdDistance, 1), 2)) * HAPtime/chargeTime, 2)
        
        if problemNo == 0: # Sum-throughput maximization (throughput의 합)
            result += throughput
        elif problemNo == 1: # Common throughput maximization problem (throughput의 최솟값)
            if result > throughput: result = throughput

        # print(throughput, chargeTime, mapSizeSquare, wdDistance)

    return result
