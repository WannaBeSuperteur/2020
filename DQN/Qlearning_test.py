# 이 파일은 Qlearning_stochastic.py 파일을 이용하여,
# Non-deterministic map에서 Q-learning을
#     1) 기본적인 방법
#     2) learning rate를 이용하여 Non-deterministic map에 보다 최적화된 방법
# 으로 수행했을 때 성공률(벽에 도달하지 않는 비율)을 비교하기 위한 파일임.

import os
import mapReader
import mapGenerator
import Qlearning_stochastic

if __name__ == '__main__':

    ### 0. config.txt 파일 읽기 ###
    
    ## config.txt 파일의 구성:
    
    # 1) line 0~(n-2): (width) (height) (N) (prob)
    # 가로 길이가 (width)이고 세로 길이가 (height)인 맵을 (N)개 생성
    # 각 맵에 대해 각 지점이 벽일 확률은 (prob)

    # 2) line (n-1): (learning rate)
    # Non-deterministic map에 적용할 learning rate
    
    file = open('config.txt', 'r')
    read = file.readlines()
    file.close

    count = 0 # 맵 파일의 이름이 testMap_XXXXXX.txt일 때 XXXXXX 부분

    ### 1. 맵을 생성하고 학습률을 지정 ###
    for i in range(len(read)):
        
        # line 0~(n-2): 맵 생성
        # 기존에 있는 파일로만 진행하려면 이 부분을 비우고 학습률만 지정
        if i < len(read)-1:
            width = int(read[i].split(' ')[0])
            height = int(read[i].split(' ')[1])
            N = int(read[i].split(' ')[2])
            prob = float(read[i].split(' ')[3])

            for j in range(N):
                fn = '{:06d}'.format(count) + '.txt'
                mapGenerator.mapGenerate(width, height, fn, prob)
                count += 1

        # line (n-1): 학습률 지정
        else:
            lrs = read[i].split(' ') # 실험할 학습률을 저장한 배열

    ### 2. testMaps 폴더에 있는 파일 이름 가져오기 ###
    pathDir = 'testMaps'
    fileList = os.listdir(pathDir)
    
    # 파일 이름에 'testMap_'이 없는 파일 제외
    for i in range(len(fileList)-1, -1, -1):
        if not 'testMap_' in fileList[i]: fileList.pop(i)

    # 결과 파일에 쓸 텍스트
    toWrite = 'FileName\tSize\tTry\tSuc(0.0)\t'

    # 학습률 출력
    lrText = '' # 학습률에 대한 정보
    for i in range(len(lrs)):
        lrText += '학습률 ' + str(lrs[i]) + ' | '
        toWrite += 'Suc(' + str(lrs[i]) + ')\t'
    toWrite += '\n'
    print('')
    print('| 파일 이름 | 크기(가로,세로) | 회차 | 기존방법 성공률 | ' + lrText)

    # 각 학습률에 대한 평균 성공률 (전체)
    avgSucTotal = []
    for k in range(len(lrs)+1): avgSucTotal.append(0)

    # testMaps 폴더에 있는 모든 파일에 대하여
    for i in range(len(fileList)):

        ### 3. 맵 읽기 ###
        fn = 'testMaps/' + fileList[i] # 파일 이름
        info = mapReader.readMap(fn, False) # 맵 읽기
        
        map_ = info[0]
        width = info[1] # 맵의 가로 길이
        height = info[2] # 맵의 세로 길이

        # 각 학습률에 대한 평균 성공률 (각 맵별)
        avgSuc = []
        for j in range(len(lrs)+1): avgSuc.append(0)

        ### 4. 본 실험 ###
        # 각 맵에 대해 총 5회의 학습률별 Q-table 갱신, 학습 및 테스트 반복
        # 맵별 총 시도횟수는 (반복횟수=5 * 시도횟수=1600) = 8000 
        for j in range(5):
            
            # 출력할 정보가 있는 텍스트
            infoText = '| ' + fn + ' | ' + str(width) + ',' + str(height) + ' | ' + str(j) + ' | '

            # 파일에 쓸 텍스트
            toWrite += fn + '\t' + str(width) + ',' + str(height) + '\t' + str(j) + '\t'
            
            ### 4-0. 기존 Q-network대로 학습 및 테스트 ###
            Qtable0 = mapReader.initQtable(width, height) # Q table 초기화
            Qlearning_stochastic.learning(map_, width, height, Qtable0, 0, 0, False)
            suc0 = Qlearning_stochastic.execute(map_, width, height, Qtable0, False) # 기존대로 했을 때의 성공횟수
            infoText += str(round(suc0/16, 1)) + '% | '
            toWrite += str(round(suc0/16, 4)) + '\t'

            # 성공횟수를 더하고 나중에 시도횟수로 나눠서 성공률을 구함
            avgSuc[0] += suc0
            avgSucTotal[0] += suc0

            ### 4-1. 학습률을 적용하여 Non-deterministic 최적화하는 방법으로 학습 및 테스트 ###
            for k in range(len(lrs)):
                Qtable = mapReader.initQtable(width, height) # Q table 초기화
                Qlearning_stochastic.learning(map_, width, height, Qtable, 1, float(lrs[k]), False)
                suc = Qlearning_stochastic.execute(map_, width, height, Qtable, False) # 학습률을 적용한 경우의 성공횟수
                infoText += str(round(suc/16, 1)) + '% | '
                toWrite += str(round(suc/16, 4)) + '\t'

                # 성공횟수를 더하고 나중에 시도횟수로 나눠서 성공률을 구함
                avgSuc[k+1] += suc
                avgSucTotal[k+1] += suc

            # 해당 시도에 대한 결과 출력
            print(infoText)
            toWrite += '\n'

        # 해당 맵에 대한 결과 출력
        # 평균 성공률의 %를 구하기 위하여 (맵별 총 시도횟수=8000 / 100) = 80으로 나눔
        for j in range(len(avgSuc)):
            avgSuc[j] /= 80
            avgSuc[j] = round(avgSuc[j], 2)
        # 각 맵의 각 학습률별 평균 성공률 출력
        print(fn + ' 평균 성공률: ' + str(avgSuc))
        print('')

    # 전체에 대한 결과 출력
    # 평균 성공률의 %를 구하기 위하여 (전체 시도횟수=(8000*맵개수) / 100) = (80*맵개수) 로 나눔
    for j in range(len(avgSuc)):
        avgSucTotal[j] /= (80 * len(fileList))
        avgSucTotal[j] = round(avgSucTotal[j], 2)
    # 각 맵의 각 학습률별 평균 성공률 출력
    print('전체 평균 성공률: ' + str(avgSucTotal))

    # 파일에 쓰기
    resultFile = open('result.txt', 'w')
    resultFile.write('<content of config.txt>\n')
    for i in range(len(read)): resultFile.write(read[i])
    resultFile.write('\n\n<test result>\n')
    resultFile.write(toWrite)
    resultFile.close()
