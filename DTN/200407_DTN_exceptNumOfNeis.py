ex = __import__('200407_DTNexample')
import deepLearning_GPU
import deepLearning_GPU_helper

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

import math
import random

# 연결 확률
def conProb(value):
    if value > 1: return 0.0
    elif value < 0: return 1.0
    else: return 0.25 + 0.75 * (1.0 - pow(value, 3))

# numOfTrain: 학습 데이터의 개수
# numOfTest : 테스트 데이터의 개수
# virusProb : next node가 malware에 감염될 확률
def mainFunc(numOfTrain, numOfTest, virusProb, boardSize, maxDist):
    count = 0
    toSave = ''

    countList = [] # 학습 및 테스트 데이터의 각 line에 대한 dataNo (데이터 번호)

    hop1Noise_ = [] # hop1Noise에 대한 전체 학습 데이터
    hop1Max_ = [] # hop1Max에 대한 전체 학습 데이터
    hop2AvgNoise_ = [] # hop2AvgNoise에 대한 전체 학습 데이터
    hop2AvgMax_ = [] # hop2AvgMax에 대한 전체 학습 데이터
    hop1Neis_ = [] # hop1Neis에 대한 전체 학습 데이터
    hop2AvgNeis_ = [] # hop2AvgNeis에 대한 전체 학습 데이터
    
    hop1NoiseZ_ = [] # hop1NoiseZ에 대한 전체 학습 데이터
    hop1MaxZ_ = [] # hop1MaxZ에 대한 전체 학습 데이터
    hop2AvgNoiseZ_ = [] # hop2AvgNoiseZ에 대한 전체 학습 데이터
    hop2AvgMaxZ_ = [] # hop2AvgMaxZ에 대한 전체 학습 데이터
    hop1NeisZ_ = [] # hop1NeisZ에 대한 전체 학습 데이터
    hop2AvgNeisZ_ = [] # hop2AvgNeisZ에 대한 전체 학습 데이터

    count_ = [] # count_에 대한 전체 데이터
    nodeNo_ = [] # nodeNo에 대한 전체 데이터
    isMal_ = [] # malicious node 여부에 대한 전체 학습 데이터

    # 학습 데이터 생성
    while count < numOfTrain + numOfTest: # numOfTrain + numOfTest개의 데이터
        try:
            print('\n ******** ******** <<< [ ' + str(count) + ' ] th training-test data >>> ******** ********\n')

            # nodeLoc   : 각 node의 3차원상의 위치
            # maxDist   : 각 node가 연결되기 위한 최대의 distance
            # DTNnetwork: 학습 데이터의 DTN 네트워크
            # malArray  : 각 node가 malicious node인지의 여부
            (nodeLoc, maxDist, DTNnetwork, malArray) = ex.makeDTN(rangeOfNodes=[20, 25], boardSize=boardSize, maxDist=maxDist, firstCenter=False, maxNeighbors=1, maxNeiCount=5, malProb=0.2)
            
            nodes = len(DTNnetwork) # DTN의 노드 개수
            nodeMove = [[0] * nodes for _ in range(nodes)] # nodeMove[A][B]: node A에서 node B로의 이동횟수

            print('<<< node location >>>')
            for i in range(nodes): print('node ' + str(i) + ' : LOC ' + str(nodeLoc[i]))
            print('')

            print('## Network ##')
            for j in range(nodes): print(DTNnetwork[j])
            print('')

            # packet 전달 300회 반복
            totalTry = 300 # 전체 try 횟수
            failedTry = 0 # failed try(목적 node로 전송 실패)의 개수
            for j in range(totalTry):
                startNode = random.randint(0, nodes-1) # 시작 노드
                destNode = random.randint(0, nodes-1) # 도착 노드

                # 시작 노드와 도착 노드가 같으면 건너뛰기
                if startNode == destNode: continue

                thisNode = startNode

                # destNode에 도착할 때까지 다음 node로 이동
                while True:

                    # DTN network 배열 복사 (단, node 간의 distance에 따라 전송 실패(배열에서 0) 확률 존재)
                    DTNnetwork_ = [[0] * nodes for _ in range(nodes)]
                    for k in range(nodes):
                        for l in range(nodes):

                            # calculate distance between node k and l
                            distance_kl = ex.distancePoint(nodeLoc[k], nodeLoc[l])

                            # calculate connection probability for the distance
                            connect_prob = conProb(distance_kl/maxDist)
                            # print(distance_kl, maxDist, connect_prob)

                            # connected or not
                            if random.random() < connect_prob: DTNnetwork_[k][l] = DTNnetwork[k][l]
                            else: DTNnetwork_[k][l] = 0

                    # 첫번째인 경우 전송가능한 node에 대한 행렬 표시 
                    if j == 0:
                        print('## Network for forwardable nodes (j == 0) ##')
                        for k in range(nodes): print(DTNnetwork_[k])
                        print('')

                    # 다음 노드 탐색
                    nextNode = ex.dijkstra_nextnode(thisNode, destNode, DTNnetwork_, malArray[thisNode], False)

                    # 일정 확률로 malicious node 바이러스 전파
                    if malArray[thisNode] == True and random.random() < float(virusProb):
                        malwareInfect = True
                        print(str(j) + '-th try: malware infection: ' + str(thisNode) + ' -> ' + str(nextNode))
                        malArray[nextNode] = True

                    # nextNode가 -1이면 startNode -> destNode의 "전송 가능한" 경로가 없다는 의미이므로 건너뛰기
                    if nextNode == -1:
                        if j < 15: print(str(j) + '-th try failed')
                        failedTry += 1
                        break

                    #print(str(j) + ' : ' + str(thisNode) + 'mal:[' + str(malArray[thisNode]) + '] -> ' + str(nextNode) + ' dest: (' + str(destNode) + ')')

                    nodeMove[thisNode][nextNode] += 1 # node A->node B에 대해 nodeMove[A, B] 갱신
                    thisNode = nextNode

                    # nextNode가 destNode와 같으면 destNode에 도착했다는 의미이므로 종료
                    if nextNode == destNode: break

            print('')
            print('try success rate: ' + str(totalTry - failedTry) + ' / ' + str(totalTry) + ' = ' + str(100 * (totalTry - failedTry)/totalTry))
            print('')
            print('## packet move from A to B ##')
            for j in range(nodes): print(nodeMove[j])
            print('')

            nodeNo = [] # node 번호 (hop1Noise의 값이 -1, 즉 이웃 node가 없는 경우 제외)

            # [A] 각 node에서 first hop neighbor로 패킷 전송 시의 noise
            # [B] 각 node에서 first hop neighbor로 패킷 전송 시의, 가장 많이 선택되는 node의 선택률
            hop1Noise = [] # [A]
            hop1Max = [] # [B]
            
            for j in range(nodes):
                hop1Noise_j = ex.noise(nodeMove[j]) # 각 node에서 first hop neighbor로 패킷 전송 시의 noise
                if hop1Noise_j >= 0:
                    nodeNo.append(j)
                    hop1Noise.append(hop1Noise_j) # [A] 데이터 추가
                    hop1Max.append(max(nodeMove[j])/sum(nodeMove[j])) # [B] 데이터 추가

            # [C] 각 node의 (그 node로부터 패킷이 전송된) first hop neighbor에서 second hop neighbor로 패킷 전송 시의 noise의 평균값
            # [D] 각 node의 (그 node로부터 패킷이 전송된) first hop neighbor에서 second hop neighbor로 패킷 전송 시의, 가장 많이 선택되는 node의 선택률의 평균값
            # [E] 각 node의 (그 node로부터 패킷이 전송된) first hop neighbor의 수 (해당 first hop neighbor에서 second hop neighbor로 패킷을 전송하지 않은 경우 제외) 
            # [F] 각 node의 각 (그 node로부터 패킷이 전송된) first hop neighbor에 대한, (해당 first hop neighbor로부터 패킷이 전송된) second hop neighbor의 수의 평균 : "[E]의 평균과 다를 수 있음"

            # 1-hop nei 중 이웃 node가 있더라도 그 이웃에 패킷을 전송하지 않은 node는 제외하고 계산
            # ex) 1-hop neighbor가 [0, 2, 8]이고 이들 모두 이웃 node가 존재. 하지만 그 이웃에 패킷을 전송한 node는 [0, 8]뿐.
            #     0, 2, 8의 [C] 값은 각각 0.0, 0.65, 1.0, [D] 값은 각각 1.0, 0.8333, 0.5
            #
            #     이때 hop1NeighborsSendPacket=2이며 [C], [D]의 값은 node 0, 8에 대한 평균으로 계산
            #     -> [C] = (0.0+1.0)/2 = 0.5, [D] = (1.0+0.5)/2 = 0.75 (not 0.55, 0.7778)
            
            hop2AvgNoise = [] # [C]
            hop2AvgMax = [] # [D]
            hop1Neis = [] # [E]
            hop2AvgNeis = [] # [F]
            
            print('\n## neighbor info (based on PACKET MOVEMENT) ##')
            for j in range(len(nodeNo)): # 각 node에 대하여 (hop1Noise의 값이 -1, 즉 이웃 node가 없는 경우 제외)

                # hop1Noise의 값이 -1이면 neighbor가 없다는 의미이므로 건너뛰기
                # hop1Noise의 값이 0 이상이면 neighbor가 있다는 의미이고, 각 neighbor에 대해서도 neighbor가 무조건 존재 (해당 node)
                if hop1Noise[j] == -1: continue

                # 패킷을 전송한 first hop neighbor를 모두 찾기
                hop1Neighbors = []
                for k in range(nodes):
                    if nodeMove[nodeNo[j]][k] > 0: hop1Neighbors.append(k)
                print(str(nodeNo[j]) + ' -> ' + str(hop1Neighbors))

                # [C], [D]의 값 계산
                hop1NeighborsSendPacket = 0 # second hop neighbor에 패킷을 전송한 first hop neighbor의 수
                
                avgNoise = 0.0 # [C] 각 (그 node로부터 패킷이 전송된) node의 first hop neighbor에서 second hop neighbor로 패킷 전송 시의 noise의 평균값
                avgMax = 0.0 # [D] 각 (그 node로부터 패킷이 전송된) node의 first hop neighbor에서 second hop neighbor로 패킷 전송 시의, 가장 많이 선택되는 node의 선택률의 평균값
                avgNeis = 0 # [F] 각 (그 node로부터 패킷이 전송된) node의 각 first hop neighbor에 대한, (해당 first hop neighbor로부터 패킷이 전송된) second hop neighbor의 수의 평균
                
                for k in range(len(hop1Neighbors)):
                    # print(str(j) + ' : ' + str(hop1Neighbors[k]) + ' -> ' + str(ex.noise(nodeMove[hop1Neighbors[k]])))
                    # print(i, j, k, nodeMove[hop1Neighbors[k]])
                    
                    if sum(nodeMove[hop1Neighbors[k]]) > 0: # 해당 first hop neighbor k가 패킷을 전송한 경우
                        avgNoise += ex.noise(nodeMove[hop1Neighbors[k]]) # [C] 데이터 추가
                        avgMax += max(nodeMove[hop1Neighbors[k]])/sum(nodeMove[hop1Neighbors[k]]) # [D] 데이터 추가
                        
                        # [F] 데이터 추가
                        for l in range(len(nodeMove[hop1Neighbors[k]])):
                            if nodeMove[hop1Neighbors[k]][l] > 0: avgNeis += 1

                        hop1NeighborsSendPacket += 1

                # [E] 데이터 추가
                hop1Neis.append(hop1NeighborsSendPacket)

                # second hop neighbor에 패킷을 전송한 first hop neighbor가 없으면 -1을 추가하기
                if hop1NeighborsSendPacket == 0:
                    hop2AvgNoise.append(-1)
                    hop2AvgMax.append(-1)
                    hop2AvgNeis.append(-1)
                else:  
                    avgNoise /= hop1NeighborsSendPacket
                    avgMax /= hop1NeighborsSendPacket
                    avgNeis /= hop1NeighborsSendPacket
                    
                    hop2AvgNoise.append(avgNoise)
                    hop2AvgMax.append(avgMax)
                    hop2AvgNeis.append(avgNeis)

                # 파일로 저장할 데이터 추가
                countList.append(count) # dataNo 값 추가
                count_.append(count)

            # 다음 배열의 크기는 모두 같아야 함
            # nodeNo, hop1Noise, hop1Max, hop2AvgNoise, hop2AvgMax, hop1Neis, hop2AvgNeis
            print('assertion...')
            assert(len(nodeNo) == len(hop1Noise) and len(hop1Noise) == len(hop1Max) and len(hop1Noise) == len(hop2AvgNoise) and len(hop1Noise) == len(hop2AvgMax) and len(hop1Noise) == len(hop1Neis) and len(hop1Noise) == len(hop2AvgNeis))
            print('assertion passed')

            # 학습 데이터에 추가
            hop1Noise_ += hop1Noise
            hop1Max_ += hop1Max
            hop2AvgNoise_ += hop2AvgNoise
            hop2AvgMax_ += hop2AvgMax
            hop1Neis_ += hop1Neis
            hop2AvgNeis_ += hop2AvgNeis

            # 파일로 저장할 데이터 추가
            nodeNo_ += nodeNo

            for i in range(len(nodeNo)): isMal_.append(malArray[nodeNo[i]])

            # 카운트 증가
            count += 1
            
        except Exception as e:
            print('\n<---- ERROR: ' + str(e) + ' ---->\n')

    # 각 node의 [A], [B], [C], [D], [E], [F]에 대한 Z 값 (학습 데이터) 생성
    
    hop1NoiseZ_ = [] # hop1Noise의 각 원소의 Z값
    hop2AvgNoiseZ_ = [] # hop2AvgNoise의 각 원소의 Z값
    hop1MaxZ_ = [] # hop1Max의 각 원소의 Z값
    hop2AvgMaxZ_ = [] # hop2AvgMax의 각 원소의 Z값
    hop1NeisZ_ = [] # hop1Neis의 각 원소의 Z값
    hop2AvgNeisZ_ = [] # hop2AvgNeis의 각 원소의 Z값
    
    for j in range(len(hop1Noise_)):
        hop1NoiseZ_.append(ex.Zvalue(hop1Noise_, hop1Noise_[j], -1))
        hop1MaxZ_.append(ex.Zvalue(hop1Max_, hop1Max_[j], -1))
        hop2AvgNoiseZ_.append(ex.Zvalue(hop2AvgNoise_, hop2AvgNoise_[j], -1))
        hop2AvgMaxZ_.append(ex.Zvalue(hop2AvgMax_, hop2AvgMax_[j], -1))
        hop1NeisZ_.append(ex.Zvalue(hop1Neis_, hop1Neis_[j], -1))
        hop2AvgNeisZ_.append(ex.Zvalue(hop2AvgNeis_, hop2AvgNeis_[j], -1))

    print('hop1Noise_=' + str(len(hop1Noise_)) + ' hop1Max_=' + str(len(hop1Max_)) + ' hop2AvgNoise_=' + str(len(hop2AvgNoise_)) + ' hop2AvgMax_=' + str(len(hop2AvgMax_)))
    print('hop1Neis_=' + str(len(hop1Neis_)) + ' hop2AvgNeis_=' + str(len(hop2AvgNeis_)))
    print('hop1NoiseZ_=' + str(len(hop1NoiseZ_)) + ' hop1MaxZ_=' + str(len(hop1MaxZ_)) + ' hop2AvgNoiseZ_=' + str(len(hop2AvgNoiseZ_)) + ' hop2AvgMaxZ_=' + str(len(hop2AvgMaxZ_)))
    print('hop1NeisZ_=' + str(len(hop1NeisZ_)) + ' hop2AvgNeisZ_=' + str(len(hop2AvgNeisZ_)))

    # -1인 데이터 제외
    for j in range(len(hop1Noise_)-1, -1, -1):
        if hop2AvgNoise_[j] == -1 or hop2AvgMax_[j] == -1:
            print('pop: ' + str(j))
            
            count_.pop(j)
            nodeNo_.pop(j)

            hop1Noise_.pop(j)
            hop1Max_.pop(j)
            hop2AvgNoise_.pop(j)
            hop2AvgMax_.pop(j)
            hop1Neis_.pop(j)
            hop2AvgNeis_.pop(j)
            
            hop1NoiseZ_.pop(j)
            hop1MaxZ_.pop(j)
            hop2AvgNoiseZ_.pop(j)
            hop2AvgMaxZ_.pop(j)
            hop1NeisZ_.pop(j)
            hop2AvgNeisZ_.pop(j)

            isMal_.pop(j)

    # [A]와 [B]의 noise data 및 Z 값 출력
    print('\n## noise data (1-hop, 2-hop avg) ##')
    print('dataNo\tnode\th1Noi\th1Max\th2Noi\th2Max\th1Nei\th2Nei\th1NoiZ\t\th1MaxZ\t\th2NoiZ\t\th2MaxZ\t\th1NeiZ\t\th2NeiZ\t\tmal?')
            
    for i in range(len(hop1Noise_)):
        toSave_ = str(count_[i]) + '\t' + str(nodeNo_[i]) + '\t' + str(round(hop1Noise_[i], 4)) + '\t' + str(round(hop1Max_[i], 4)) + '\t'
        toSave_ += str(round(hop2AvgNoise_[i], 4)) + '\t' + str(round(hop2AvgMax_[i], 4)) + '\t'
        toSave_ += str(hop1Neis_[i]) + '\t' + str(round(hop2AvgNeis_[i], 4)) + '\t'
        
        toSave_ += str(round(hop1NoiseZ_[i], 10)) + '\t' + str(round(hop1MaxZ_[i], 10)) + '\t'
        toSave_ += str(round(hop2AvgNoiseZ_[i], 10)) + '\t' + str(round(hop2AvgMaxZ_[i], 10)) + '\t'
        toSave_ += str(round(hop1NeisZ_[i], 10)) + '\t' + str(round(hop2AvgNeisZ_[i], 10)) + '\t' + str(isMal_[i])

        print(toSave_)
        toSave += toSave_ + '\n'

    # 파일로 저장
    print('[', len(toSave))
    f = open('DTN_result.txt', 'w')
    f.write(toSave)
    f.close()

    f = open('DTN_result_args.txt', 'w')
    f.write(str(numOfTrain) + ' ' + str(numOfTest) + ' ' + str(virusProb) + '\nboardSize=' + str(boardSize) + ' maxDist=' + str(maxDist))
    f.close()

    # 결과 반환
    return [[hop1Noise_, hop2AvgNoise_, hop2AvgNoise_, hop2AvgMax_, hop1Neis_, hop2AvgNeis_,
            hop1NoiseZ_, hop1MaxZ_, hop2AvgNoiseZ_, hop2AvgMaxZ_, hop1NeisZ_, hop2AvgNeisZ_, isMal_], countList]

def trainAndTest(numOfTrain, numOfTest, virusProb, boardSize, maxDist, deviceName): # 학습 데이터를 읽어서 배열에 저장 및 학습

    # 학습 및 테스트 데이터 파일 존재 여부와 무관하게 실행
    # 학습 데이터를 mainFunc을 이용하여 생성
    mF = mainFunc(numOfTrain, numOfTest, virusProb, boardSize, maxDist)
        
    mainData = mF[0] # 학습 및 테스트 데이터의 각 line
    countList = mF[1] # 파일의 각 line에 대한 데이터 번호 (dataNo)

    # 학습 실시
    #print(inputs[0:5])
    epochList = input('epochs:')
    testResult = ''
    testTable = 'epoch\ttrain\ttest\terror\tTP\tTPrate\tTN\tTNrate\tFP\tFPrate\tFN\tFNrate\treal P\trP rate\treal N\trN rate\tcorrect\tco rate\n'
    
    for epoch in range(len(epochList.split(' '))):

        # 학습 데이터 (각 x열에 해당하는 8개의 배열, y열에 해당하는 배열)
        # 0 ~ 5 : hop1Noise_, hop2AvgNoise_, hop2AvgNoise_, hop2AvgMax_, hop1Neis_, hop2AvgNeis_
        # 6 ~11 : hop1NoiseZ_, hop1MaxZ_, hop2AvgNoiseZ_, hop2AvgMaxZ_, hop1NeisZ_, hop2AvgNeisZ_,
        #  12   : isMal_
        mainData_train = [[], [], [], [], [], [], [], [], [], [], [], [], []] # 학습 데이터
        mainData_test = [[], [], [], [], [], [], [], [], [], [], [], [], []] # 테스트 데이터

        # 각 데이터를 랜덤하게 학습데이터(0) 또는 테스트데이터(1) 로 지정
        dataInfo = []
        dataSize = numOfTrain + numOfTest # 전체 데이터의 개수
        for i in range(dataSize): dataInfo.append(1)

        trainCount = 0
        
        while trainCount < numOfTrain: # 학습데이터(0) 지정
            index = math.floor(random.random() * dataSize)
            if dataInfo[index] != 0: # 기존에 학습데이터로 지정되지 않음
                trainCount += 1
                dataInfo[index] = 0

        # 숫자 데이터를 float으로 변환
        for i in range(len(mainData)):
            for j in range(12): mainData[j][i] = float(mainData[j][i])

        # numOfTrain_NB, numOfTrain_SVM, numOfTest에 추가
        for i in range(len(mainData[0])):
            dataNoOfThisLine = countList[i] # 데이터 파일의 해당 line의 dataNo 값
                
            if dataInfo[dataNoOfThisLine] == 0: # 0 -> 학습 데이터
                for j in range(13): mainData_train[j].append(mainData[j][i])
            elif dataInfo[dataNoOfThisLine] == 1: # 1 -> 테스트 데이터
                for j in range(13): mainData_test[j].append(mainData[j][i])
                    
        # 학습 데이터, 테스트 데이터의 실제 line 개수 (각 데이터마다 node 개수만큼의 line이 있음)
        lenTrain = len(mainData_train[0])
        lenTest = len(mainData_test[0])
        print('length:', lenTrain, lenTest)

        print('\n################################')
        print(lenTrain, lenTest, virusProb, 'boardSize:', boardSize, 'maxDist:', maxDist)
        print('dataInfo: ' + str(dataInfo))
        print('################################\n')

        # isMal_의 'True', 'False'를 숫자 0(False), 1(True)로 변환
        for i in range(lenTrain):
            if mainData_train[12][i] == True: mainData_train[8][i] = 1
            else: mainData_train[12][i] = 0
        for i in range(lenTest):
            if mainData_test[12][i] == True: mainData_test[8][i] = 1
            else: mainData_test[12][i] = 0

        # 신경망 생성
        NN = [tf.keras.layers.Flatten(input_shape=(4,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')]
        op = tf.keras.optimizers.Adam(0.001)

        # 학습 데이터의 입력, 출력 데이터 생성
        # 입력 데이터: hop1NoiseZ_, hop1MaxZ_, hop2AvgNoiseZ_, hop2AvgMaxZ_(6~9) / 출력 데이터: isMal_(12)
        inputs = []
        outputs = []
        for i in range(lenTrain):
            inputs.append([mainData_train[6][i], mainData_train[7][i], mainData_train[8][i], mainData_train[9][i]])
            outputs.append([mainData_train[12][i]])

        # 테스트 데이터의 입력 데이터 생성
        testInputs = [] # 테스트 입력
        testValOutputs = [] # 테스트에 대한 validation data (실제 결과값)
        for i in range(lenTest):
            testInputs.append([mainData_test[6][i], mainData_test[7][i], mainData_test[8][i], mainData_test[9][i]])
            testValOutputs.append(mainData_test[12][i])

        # 학습 실시
        ep = int(epochList.split(' ')[epoch])
        deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', inputs, outputs, '200407_DTN', ep, False, True, deviceName)

        # 테스트 실시
        testModel = deepLearning_GPU.deepLearningModel('200407_DTN', True)
        testOutputAllLayers = deepLearning_GPU.modelOutput(testModel, testInputs)
        testOutput = testOutputAllLayers[len(testOutputAllLayers)-1] # 모델에 testInputs를 입력했을 때의 결과값

        # 오차 제곱의 평균 계산
        difAvg = 0
        for i in range(lenTest): difAvg += pow(testOutput[i][0] - testValOutputs[i], 2)
        difAvg /= lenTest
        print('mean square error: ' + str(difAvg))
        print('')

        # TP, TN, FP, FN 계산
        TP = 0 # True Positive
        TN = 0 # True Negative
        FP = 0 # False Positive
        FN = 0 # False Negative

        for i in range(lenTest):
            if testOutput[i][0] >= 0.5 and testValOutputs[i] >= 0.5: TP += 1
            elif testOutput[i][0] < 0.5 and testValOutputs[i] < 0.5: TN += 1
            elif testOutput[i][0] >= 0.5 and testValOutputs[i] < 0.5: FP += 1
            elif testOutput[i][0] < 0.5 and testValOutputs[i] >= 0.5: FN += 1

        ALL = TP + TN + FP + FN
        RealP = TP + FN
        RealN = TN + FP
        correct = TP + TN

        # 테스트 결과 저장
        testResult += ' <<< epoch ' + str(ep) + ' >>>\n'
        testResult += 'mean square error: ' + str(difAvg) + ' (train: ' + str(len(inputs)) + ' test:' + str(ALL) + ')\n\n'
        
        testResult += 'True Positive    : ' + str(TP) + ' / ' + str(round(100 * TP / ALL, 2)) + '%\n'
        testResult += 'True Negative    : ' + str(TN) + ' / ' + str(round(100 * TN / ALL, 2)) + '%\n'
        testResult += 'False Positive   : ' + str(FP) + ' / ' + str(round(100 * FP / ALL, 2)) + '%\n'
        testResult += 'False Negative   : ' + str(FN) + ' / ' + str(round(100 * FN / ALL, 2)) + '%\n\n'

        testResult += 'Real Positive    : ' + str(RealP) + ' / ' + str(round(100 * RealP / ALL, 2)) + '%\n'
        testResult += 'Real Negative    : ' + str(RealN) + ' / ' + str(round(100 * RealN / ALL, 2)) + '%\n'
        testResult += 'correct          : ' + str(correct) + ' / ' + str(round(100 * correct / ALL, 2)) + '%\n\n'

        testTable += str(ep) + '\t' + str(len(inputs)) + '\t' + str(ALL) + '\t' + str(round(difAvg, 4)) + '\t' + str(TP) + '\t' + str(round(100 * TP / ALL, 2)) + '%\t'
        testTable += str(TN) + '\t' + str(round(100 * TN / ALL, 2)) + '%\t'
        testTable += str(FP) + '\t' + str(round(100 * FP / ALL, 2)) + '%\t'
        testTable += str(FN) + '\t' + str(round(100 * FN / ALL, 2)) + '%\t'
        testTable += str(RealP) + '\t' + str(round(100 * RealP / ALL, 2)) + '%\t'
        testTable += str(RealN) + '\t' + str(round(100 * RealN / ALL, 2)) + '%\t'
        testTable += str(correct) + '\t' + str(round(100 * correct / ALL, 2)) + '%\n'

    # 파일에 쓰기
    tR = open('DTN_testResult.txt', 'w')
    tR.write(testResult)
    tR.close()

    tT = open('DTN_testTable.txt', 'w')
    tT.write(testTable)
    tT.close()

if __name__ == '__main__':
    numOfTrain = int(input('number of train datasets'))
    numOfTest = int(input('number of test datasets'))
    virusProb = float(input('probability of malware virus infection of target node'))
    boardSize = float(input('board Size'))
    maxDist = float(input('maximum distance between 2 nodes to connect them'))
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    
    trainAndTest(numOfTrain, numOfTest, virusProb, boardSize, maxDist, deviceName)
