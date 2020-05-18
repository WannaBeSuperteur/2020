import math
import os
import random
import mapReader

### 0. 학습 ###
def learning(map_, width, height, Qtable, d, isPrint):
    state = [0, 0] # 현재 상태 (좌표: 행렬에서와 같이 [세로좌표, 가로좌표] 형식)
    SARS_set = [] # 학습 종료 후 반환할 (state, action, reward, state+1) 데이터

    # 반복횟수(iters) 지정
    if d == 0: iters = 20000
    elif d == 1: iters = 5000

    # 20000번 반복
    for iteration in range(iters):
        # 정보 출력
        if isPrint:
            # d=0 -> 다음 상태를 현재 상태에서 행동을 취한 이후의 상태로 갱신
            if d == 0 and (iteration < 50 or (iteration < 1000 and iteration % 50 == 0) or iteration % 500 == 0 or iteration == 19999):
                print('<' + str(iteration) + ' 회 학습 후>')
                printMap(map_, width, height, state, Qtable)
            # d=1 -> 다음 상태를 랜덤으로 갱신
            elif d == 1 and (iteration < 75 or (iteration < 1500 and iteration % 50 == 0) or iteration % 500 == 0 or iteration == 4999):
                print('<' + str(iteration) + ' 회 학습 후>')
                printMap(map_, width, height, state, Qtable)

        # 목표 상태가 아니면 행동 탐색
        if map_[state[0]][state[1]] != 'G':

            ## 0-0. 각 행동이 실행 가능한지 여부 파악 ##
            feasible = isFeasible(map_, state) # 상, 하, 좌, 우 각 방향의 이동 가능 여부

            ## 0-1. 현재 상태 s에서 행동 a를 실행
            decided = -1 # 실행할 행동의 번호 (0:상, 1:하, 2:좌, 3:우로 이동)

            # 모든 action에 대한 보상값이 0이면 무조건 랜덤하게 실행
            allZero = True # 현재 상태에서 가능한 모든 action에 대한 보상이 0인가?
            for i in range(4):
                if Qtable[state[0]][state[1]][i] > 0.0:
                    allZero = False
                    break

            # 랜덤하게 행동 실행
            if random.random() < randomAction(iteration) or allZero:
                while(1):
                    randAction = random.randint(0, 3) # 랜덤하게 실행할 행동의 번호
                    if feasible[randAction]:
                        decided = randAction
                        break

            # 최적의 행동 실행
            else:
                for i in range(4): # 0~3의 행동 중 처음으로 실행 가능한 행동을 찾아서 실행
                    if feasible[i]:
                        decided = i
                        break

            ## 0-2. 새로운 상태를 정의 ##
            newState = [-1, -1] # 새로운 상태를 나타내는 좌표
            if decided == 0: newState = [state[0]-1, state[1]] # 상
            elif decided == 1: newState = [state[0]+1, state[1]] # 하
            elif decided == 2: newState = [state[0], state[1]-1] # 좌
            elif decided == 3: newState = [state[0], state[1]+1] # 우
            
            ## 0-3. 보상 받기 ##
            # 목표 지점에 도달하면 9.9의 보상을 받음
            if map_[newState[0]][newState[1]] == 'G':
                reward = 9.9
            else:
                reward = 0.0

            ## 0-4. 새로운 상태에서 취할 수 있는 행동 탐색 ##
            maxReward = 0.0 # 그 행동에 대한 보상 중 가장 큰 값
            for i in range(4):
                if Qtable[newState[0]][newState[1]][i] > maxReward:
                    maxReward = Qtable[newState[0]][newState[1]][i]

            ## 0-5. 현재 상태 s에서 행동 decided를 취할 때의 보상값을 업데이트 ##
            Qtable[state[0]][state[1]][decided] = reward + 0.95 * maxReward

            # 학습 데이터에 추가
            SARS_set.append([[state[0], state[1]], decided, reward + 0.95 * maxReward, newState])

        ## 0-6. 현재 상태 갱신 ##
        # 목표 지점에 도달한 경우 (0, 0)으로, 그렇지 않은 경우 newState로
        if map_[state[0]][state[1]] == 'G':
            state = [0, 0]
        else:
            if d == 0: # 새로운 상태를 현재 상태의 이웃으로 갱신
                state = [newState[0], newState[1]]
            else: # 새로운 상태를 랜덤으로 갱신
                while(1):
                    tempX = random.randint(0, width-1)
                    tempY = random.randint(0, height-1)
                    if not map_[tempY][tempX] == 'G' and not map_[tempY][tempX] == '1':
                        state = [tempY, tempX]
                        break

    # (state, action, reward, state+1) 데이터 반환
    return SARS_set
        
### 1. 실행 ###
def execute(map_, width, height, Qtable, isPrint):
    state = [0, 0] # 현재 상태
    count = 0 # 이동 횟수

    # 목표 지점에 도달할 때까지 반복
    while map_[state[0]][state[1]] != 'G':
        # 정보 출력
        if isPrint:
            print('<' + str(count) + ' 회 이동 후>')
            printMap(map_, width, height, state, Qtable)
        
        ## 1-0. 각 행동이 실행 가능한지 여부 파악 ##
        feasible = isFeasible(map_, state) # 상, 하, 좌, 우 각 방향의 이동 가능 여부
        
        ## 1-1. 현재 상태에서 취할 수 있는 행동 탐색 ##
        decided = -1 # 실행할 행동의 번호
        maxReward = 0 # 현재 상태에서 취할 수 있는 행동들에 대한 가장 큰 보상값
        for i in range(4):
            if feasible[i] == False: continue # 가능하지 않은 행동은 건너뛰기

            if Qtable[state[0]][state[1]][i] > maxReward:
                maxReward = Qtable[state[0]][state[1]][i]
                decided = i

        ## 1-2. 행동을 취하고 현재 상태 갱신 ##
        if decided == 0: state = [state[0]-1, state[1]] # 상
        elif decided == 1: state = [state[0]+1, state[1]] # 하
        elif decided == 2: state = [state[0], state[1]-1] # 좌
        elif decided == 3: state = [state[0], state[1]+1] # 우

        count += 1

    if isPrint: print('목표 지점 도달')

### 2. 기타 함수 ###
## 2-0. 랜덤으로 행동을 취할 확률에 대한 함수: 5000 / (5000 + 반복횟수) ##
def randomAction(iteration):
    return 5000 / (5000 + iteration)

## 2-1. 맵 출력 함수 ##
def rewardPrint(reward):
    if reward == 0.0:
        return ' - '
    else:
        return str(round(reward, 1))
        
def printMap(map_, width, height, state, Qtable):
    for i in range(height):
        print('*---------'*width + '*')

        # 첫째 줄
        printText = ''
        for j in range(width):
            if map_[i][j] == '1' or map_[i][j] == 'G':
                printText += ('|         ')
            elif state[0] == i and state[1] == j:
                printText += ('|***' + rewardPrint(Qtable[i][j][0]) + '***')
            else:
                printText += ('|   ' + rewardPrint(Qtable[i][j][0]) + '   ')
        printText += '|'
        print(printText)

        # 둘째 줄
        printText = ''
        for j in range(width):
            if map_[i][j] == '1':
                printText += ('|         ')
            elif map_[i][j] == 'G':
                printText += ('|    G    ')
            elif state[0] == i and state[1] == j:
                printText += ('|' + rewardPrint(Qtable[i][j][2]) + '***' + rewardPrint(Qtable[i][j][3]))
            elif map_[i][j] == 'S':
                printText += ('|' + rewardPrint(Qtable[i][j][2]) + ' S ' + rewardPrint(Qtable[i][j][3]))
            else:
                printText += ('|' + rewardPrint(Qtable[i][j][2]) + '   ' + rewardPrint(Qtable[i][j][3]))
        printText += '|'
        print(printText)

        # 셋째 줄
        printText = ''
        for j in range(width):
            if map_[i][j] == '1' or map_[i][j] == 'G':
                printText += ('|         ')
            elif state[0] == i and state[1] == j:
                printText += ('|***' + rewardPrint(Qtable[i][j][1]) + '***')
            else:
                printText += ('|   ' + rewardPrint(Qtable[i][j][1]) + '   ')
        printText += '|'
        print(printText)
        
    print('*---------'*width + '*')

## 2-2. 각 행동이 실행 가능한지 여부를 파악하는 배열을 반환하는 함수
def isFeasible(map_, state):
    feasible = [True, True, True, True] # 상, 하, 좌, 우 이동이 실행 가능한가?

    height = len(map_)
    width = len(map_[0])

    # 상
    if state[0] <= 0: feasible[0] = False
    elif map_[state[0]-1][state[1]] == '1': feasible[0] = False
    # 하
    if state[0] >= height-1: feasible[1] = False
    elif map_[state[0]+1][state[1]] == '1': feasible[1] = False
    # 좌
    if state[1] <= 0: feasible[2] = False
    elif map_[state[0]][state[1]-1] == '1': feasible[2] = False
    # 우
    if state[1] >= width-1: feasible[3] = False
    elif map_[state[0]][state[1]+1] == '1': feasible[3] = False

    return feasible

if __name__ == '__main__':
    ### 3. 기본정보 및 하이퍼파라미터 입력 ###
    fn = input('파일 이름 입력(예: map.txt):') # 파일 이름 입력
    isPrint = True # 출력 여부
    
    (map_, width, height) = mapReader.readMap(fn, True) # 맵 읽기
    Qtable = mapReader.initQtable(width, height) # Q table 초기화

    ### 4. 사용자 입력 ###
    while(1):
        print('')
        c = int(input('옵션 입력(0: 학습, 1: 실행, 2: 종료):'))
        if c == 0:
            d = int(input('학습방법 입력(0: 다음상태를 현재상태에서 행동을 취한 상태로, 1: 다음상태를 맵에서 랜덤으로 지정'))
            learning(map_, width, height, Qtable, d, isPrint)
        elif c == 1: execute(map_, width, height, Qtable, isPrint)
        elif c == 2: break
