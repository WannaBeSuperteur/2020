import os
import random

### 0. 시작 지점에서 목표 지점에 도달할 수 있는지 검사 ###
# 단, 시작지점은 왼쪽 위 끝, 목표지점은 오른쪽 아래 끝에 있음
# 
def isReachable(map_, prob):
    # 규칙:
    # 0: 이동가능한 지점, 1: 벽, 2: 이동가능하다고 확인된 지점으로 검사 불필요

    width = len(map_[0]) # 맵의 가로 길이
    height = len(map_) # 맵의 세로 길이
    
    state = [0, 0] # 현재 좌표
    queue = [] # 현재 좌표에서 이동 가능한 좌표를 저장하는 배열

    ## 0-0. 맵 복사
    map__ = [] # 복사된 맵
    for i in range(height):
        temp = [] # 맵의 각 가로줄
        for j in range(width):
            temp.append(map_[i][j])
        map__.append(temp)
        
    map__[0][0] = '2' # 시작지점을 이동가능하다고 확인된 지점으로 지정
    map__[height-1][width-1] = '0' # 끝 지점을 이동가능한 지점으로 지정

    while(1):
        ## 0-1. 상하좌우 1칸씩 이동한 지점이 이동 가능한지 확인 및
              #현재 좌표에서 이동 가능한 좌표들을 저장
        if state[0] > 0: # 1칸 위 지점으로 이동 가능?
            up1block = map__[state[0]-1][state[1]] # 1칸 위 지점
            if up1block == '0': queue.append([state[0]-1, state[1]])

        if state[0] < height-1: # 1칸 아래 지점으로 이동 가능?
            down1block = map__[state[0]+1][state[1]] # 1칸 위 지점
            if down1block == '0': queue.append([state[0]+1, state[1]])

        if state[1] > 0: # 1칸 왼쪽 지점으로 이동 가능?
            left1block = map__[state[0]][state[1]-1] # 1칸 왼쪽 지점
            if left1block == '0': queue.append([state[0], state[1]-1])

        if state[1] < width-1: # 1칸 오른쪽 지점으로 이동 가능?
            right1block = map__[state[0]][state[1]+1] # 1칸 오른쪽 지점
            if right1block == '0': queue.append([state[0], state[1]+1])

        ## 0-2. 현재 좌표가 목표 지점이면 True를 반환
        if state[0] == height-1 and state[1] == width-1: return True

        ## 0-3. 현재 좌표를 이동 가능하다고 확인된 좌표로 지정
        map__[state[0]][state[1]] = '2'

        ## 0-4. queue에서 이동 가능한 좌표를 꺼내서 그 좌표로 이동
        if len(queue) > 0:
            newState = queue.pop(0)
            state = [newState[0], newState[1]]
        else: break # queue가 비어 있으면 종료

    ## 0-5. queue가 비어 있게 될 때까지 목표 지점을 찾지 못했으면 False를 반환
    if prob <= 0.6: # 확률이 0.6 이하인 경우 목표 도달불가 맵정보 표시
        print('목표 지점 도달 불가:')
        for i in range(height):
            printText = '' # 목표 지점 도달 불가능한 맵의 각 가로줄
            for j in range(width):
                printText += ' ' + map__[i][j] + ' '
            print(printText)
        return False

### 1. 맵 생성 함수 ###
# width: 맵의 가로길이, height: 맵의 세로길이
# fn: 파일 이름, prob: 맵의 각 지점이 벽일 확률
def mapGenerate(width, height, fn, prob):

    # 벽일 확률을 최대 80%까지 설정 가능
    if prob > 0.8:
        print('벽일 확률은 최대 80%까지 지정할 수 있습니다.')
        return

    # 시작 지점에서 목표 지점에 도달할 수 있는 맵을 만들 때까지 반복
    while(1):

        ## 1-0. 맵을 표현하기 위한 배열 생성 ##
        map_ = []
        for i in range(height):
            temp = [] # 맵의 각 가로줄
            for j in range(width):
                temp.append('0') # 맵의 각 지점을 0으로 초기화 
            map_.append(temp)
    
        ## 1-1. 확률에 따라 랜덤으로 벽 지정 ##
        for i in range(height):
            for j in range(width):
                if random.random() < prob: map_[i][j] = '1'

        map_[0][0] = 'S' # 시작 지점(왼쪽 위 끝)
        map_[height-1][width-1] = 'G' # 목표 지점(오른쪽 아래 끝)

        ## 1-2. 시작 지점에서 목표 지점에 도달할 수 있는지 검사 ##
        if isReachable(map_, prob): break # 도달할 수 있으면 종료

    ## 1-3. 맵 배열을 텍스트 정보(각각의 가로줄)들로 바꾸고 그것을 파일에 쓰기 ##
    file = open('testMaps/testMap_' + fn, 'w') # 맵 파일 생성
    for i in range(height):
        oneLine = '' # 각각의 가로줄
        for j in range(width): oneLine += map_[i][j]
        if i < height-1: oneLine += '\n' # 개행문자 추가
        file.write(oneLine)
    file.close

### 2. 맵 생성
if __name__ == '__main__':
    # 4x4 크기의 벽일 확률이 0.2인 맵 10개 생성
    for i in range(10): mapGenerate(4, 4, ('0' if i < 10 else '') + str(i) + '.txt', 0.2)
    # 5x5 크기의 벽일 확률이 0.2인 맵 10개 생성
    for i in range(10, 20): mapGenerate(5, 5, ('0' if i < 10 else '') + str(i) + '.txt', 0.2)
    # 6x6 크기의 벽일 확률이 0.2인 맵 10개 생성
    for i in range(20, 30): mapGenerate(6, 6, ('0' if i < 10 else '') + str(i) + '.txt', 0.2)
