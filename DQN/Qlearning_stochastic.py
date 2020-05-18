import math
import os
import random
import mapReader

### 행동 실행 함수 ###
def makeAction(width, height, decided, state):

    # 50% 확률로 그 행동을 그대로 실행
    if random.random() < 0.5:
        if decided == 0: newState = [state[0]-1, state[1]] # 상
        elif decided == 1: newState = [state[0]+1, state[1]] # 하
        elif decided == 2: newState = [state[0], state[1]-1] # 좌
        elif decided == 3: newState = [state[0], state[1]+1] # 우

    # 50% 확률로 다른 행동 실행 (단, 맵 경계를 벗어날 수 없음)
    else:
        # 상하좌우 이동(0~3), 그대로(4) 각각이 맵의 경계 안에 위치하는가?
        inBoundary = [True, True, True, True, True]
        
        if state[0] <= 0: inBoundary[0] = False
        if state[0] >= height-1: inBoundary[1] = False
        if state[1] <= 0: inBoundary[2] = False
        if state[1] >= width-1: inBoundary[3] = False

        while(1):
            changedDecision = random.randint(0, 4) # 바뀐 행동 (0:상, 1:하, 2:좌, 3:우, 4:그대로)

            # 바뀐 행동을 실행한 결과 맵 경계를 벗어나지 않는다면 실행
            if inBoundary[changedDecision]:
                if changedDecision == 0: newState = [state[0]-1, state[1]] # 상
                elif changedDecision == 1: newState = [state[0]+1, state[1]] # 하
                elif changedDecision == 2: newState = [state[0], state[1]-1] # 좌
                elif changedDecision == 3: newState = [state[0], state[1]+1] # 우
                elif changedDecision == 4: newState = [state[0], state[1]] # 그대로
                break

    return newState
    

### 0. 학습 ###
def learning(map_, width, height, Qtable, d, lr, isPrint):
    state = [0, 0] # 현재 상태 (좌표: 행렬에서와 같이 [세로좌표, 가로좌표] 형식)
    SARS_set = [] # 학습 종료 후 반환할 (state, action, reward, state+1) 데이터

    # 100000번 반복
    for iteration in range(100000):
        # 정보 출력
        if isPrint and (iteration < 50 or iteration % 5000 == 0 or iteration == 99999):
            print('<' + str(iteration) + ' 회 학습 후>')
            printMap(map_, width, height, state, Qtable)

        # 목표 상태가 아니면 행동 탐색
        if map_[state[0]][state[1]] != 'G' and map_[state[0]][state[1]] != '1':

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
            newState = makeAction(width, height, decided, state) # 새로운 상태를 나타내는 좌표
            
            ## 0-3. 보상 받기 ##
            # 목표 지점에 도달하면 9.9의 보상을 받음
            # '1', 즉 벽에 도달하면 -9.9의 보상을 받음
            if map_[newState[0]][newState[1]] == 'G':
                reward = 9.9
            elif map_[newState[0]][newState[1]] == '1':
                reward = -9.9
            else:
                reward = 0.0

            ## 0-4. 벽에 도달하지 않았으면 새로운 상태에서 취할 수 있는 행동 탐색 ##
            maxReward = 0.0 # 그 행동에 대한 보상 중 가장 큰 값
            if map_[newState[0]][newState[1]] != '1':
                for i in range(4):
                    if Qtable[newState[0]][newState[1]][i] > maxReward:
                        maxReward = Qtable[newState[0]][newState[1]][i]

            ## 0-5. 현재 상태 s에서 행동 decided를 취할 때의 보상값을 업데이트, 학습률은 lr ##
            if d == 0: Qtable[state[0]][state[1]][decided] = reward + 0.95 * maxReward
            else: Qtable[state[0]][state[1]][decided] = (1-lr) * Qtable[state[0]][state[1]][decided] + lr * (reward + 0.95 * maxReward)

            # 학습 데이터에 추가
            SARS_set.append([[state[0], state[1]], decided, reward + 0.95 * maxReward, newState])

        ## 0-6. 현재 상태 갱신 ##
        # 목표 지점에 도달했거나 벽에 도달한 경우 (0, 0)으로, 그렇지 않은 경우 newState로
        if map_[state[0]][state[1]] == 'G' or map_[state[0]][state[1]] == '1':
            state = [0, 0]
        else:
            state = [newState[0], newState[1]]

    # (state, action, reward, state+1) 데이터 반환
    return SARS_set
        
### 1. 실행 ###
def execute(map_, width, height, Qtable, isPrint):
    success = 0 # 성공 (목표 지점 도달) 횟수

    # 목표 지점에 도달할 때까지 반복
    for i in range(1600): # 1600회 시도로 성공률 판정

        # 100회마다 성공률 표시
        if isPrint and i > 0 and i % 100 == 0:
            print('성공률: ' + str(success) + '/' + str(i) + ' = ' + str(round(success/(i/100), 2)) + '%')

        state = [0, 0] # 현재 상태
        succeed = True # 성공 여부
        count = 0 # 이동 횟수
        
        while map_[state[0]][state[1]] != 'G':
            # 처음 3회 시도에 대해서만 정보 출력
            if isPrint and i < 3:
                print('<' + str(i) + '회차 시도 ' + str(count) + ' 회 이동 후>')
                printMap(map_, width, height, state, Qtable)
            
            ## 1-0. 각 행동이 실행 가능한지 여부 파악 ##
            feasible = isFeasible(map_, state) # 상, 하, 좌, 우 각 방향의 이동 가능 여부
            
            ## 1-1. 현재 상태에서 취할 수 있는 행동 탐색 ##
            decided = -1 # 실행할 행동의 번호
            maxReward = -999999 # 현재 상태에서 취할 수 있는 행동들에 대한 가장 큰 보상값
            for j in range(4):
                if feasible[j] == False: continue # 가능하지 않은 행동은 건너뛰기

                if Qtable[state[0]][state[1]][j] > maxReward:
                    maxReward = Qtable[state[0]][state[1]][j]
                    decided = j

            ## 1-2. 행동을 취하고 현재 상태 갱신 ##
            newState = makeAction(width, height, decided, state) # 새로운 상태를 나타내는 좌표
            state = [newState[0], newState[1]]

            ## 1-3. 갱신된 현재 상태가 벽이면 실패 처리 ##
            if map_[state[0]][state[1]] == '1':
                succeed = False
                break

            count += 1

        # 성공률 측정 (처음 50회에 대해서만 표시)
        if succeed:
            success += 1
            if isPrint and i < 50: print('시도 ' + str(i) + ': success')
        else:
            if isPrint and i < 50: print('시도 ' + str(i) + ': fail')

    if isPrint: print('성공률: ' + str(success) + '/1600 = ' + str(round(success/16, 2)) + '%')
    return success # 성공 횟수 반환

### 2. 기타 함수 ###
## 2-0. 랜덤으로 행동을 취할 확률에 대한 함수: 25000 / (25000 + 반복횟수) ##
def randomAction(iteration):
    return 25000 / (25000 + iteration)

## 2-1. 맵 출력 함수 ##
def rewardPrint(reward):
    result = round(reward, 1)
    if result == 0.0:
        return ' - '
    elif result > 0:
        return str(round(result, 1))
    elif result <= -99.5:
        return '-in'
    elif result <= -9.5:
        return str(int(round(result, 0)))
    elif result <= -1:
        return str(int(round(result, 0))) + '.'
    else:
        return '-.'+str(int(round(result*(-10), 0)))
        
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
    lr = float(input('학습률 입력:')) # 학습률 입력
    isPrint = True # 출력 여부

    (map_, width, height) = mapReader.readMap(fn, True) # 맵 읽기
    Qtable = mapReader.initQtable(width, height) # Q table 초기화

    ### 4. 사용자 입력 ###
    while(1):
        print('')
        c = int(input('옵션 입력(0: 학습, 1: 실행, 2: 종료):'))
        if c == 0:
            d = int(input('학습방법 입력(0: 기존 Q-network대로 학습, 1: 학습률을 적용하여 Non-deterministic에 최적화)'))
            learning(map_, width, height, Qtable, d, lr, isPrint)
        elif c == 1: execute(map_, width, height, Qtable, isPrint)
        elif c == 2: break
