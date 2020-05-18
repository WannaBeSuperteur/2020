# 파일로부터 맵 읽기
def readMap(fn, isPrint):
    ## 0. Frozen Lake를 위한 맵이 저장된 파일(map.txt) 읽기 ##
    file = open(fn, 'r') # Frozen Lake의 맵이 저장된 파일
    map_ = file.readlines() # Frozen Lake의 맵을 나타낸 배열
    file.close

    ## 1. 각 줄(map_[i])마다 오른쪽에 있는 개행문자 제거 ##
    for i in range(len(map_)):
        map_[i] = map_[i].replace('\n', '')

    ## 2. 맵에 대한 정보 출력 ##
    width = len(map_[0]) # 맵의 가로 길이
    height = len(map_) # 맵의 세로 길이

    if isPrint:
        print('맵의 가로 길이:', width, '세로 길이:', height)
        print('맵 정보:')
        for i in range(len(map_)):
            # 각 가로줄을 나타내는 printText를 출력한다.
            printText = ''
            for j in range(len(map_[i])): # 해당 가로줄에 있는 각 지점에 대하여
                if map_[i][j] == 'S' or map_[i][j] == 'G': # S(시작) 또는 G(목표) 이면
                    printText += ('<' + map_[i][j] + '>') # 강조 표시
                else:
                    printText += (' ' + map_[i][j] + ' ') # 그렇지 않으면 표시 없음
            print(printText)

    ## 3. (맵), (가로 길이), (세로 길이) 반환
    return (map_, width, height)

# Q table 초기화
def initQtable(width, height):
    # (Q-table의 shape: [height, width, 4], 참조 시 Qtable[세로좌표][가로좌표][행동])
    Qtable = [] # Q-table (상태와 행동에 대한 보상값)
    for i in range(height):
        temp = [] # 맵의 각 가로줄에 대해 상태 및 행동에 대한 보상값 초기화
        for j in range(width):
            # 4개의 행동은 각각 상, 하, 좌, 우로 이동
            temp.append([0.0, 0.0, 0.0, 0.0]) # 0으로 초기화
        Qtable.append(temp)

    return Qtable
