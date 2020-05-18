ex = __import__('200331_DTNexample')
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
    else: return 1.0 - value

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
    
    hop1NoiseZ_ = [] # hop1NoiseZ에 대한 전체 학습 데이터
    hop1MaxZ_ = [] # hop1MaxZ에 대한 전체 학습 데이터
    hop2AvgNoiseZ_ = [] # hop2AvgNoiseZ에 대한 전체 학습 데이터
    hop2AvgMaxZ_ = [] # hop2AvgMaxZ에 대한 전체 학습 데이터
    
    isMal_ = [] # malicious node 여부에 대한 전체 학습 데이터

    # 학습 데이터 생성
    while count < numOfTrain + numOfTest: # numOfTrain + numOfTest개의 데이터
        try:
            print('\n ******** ******** <<< [ ' + str(count) + ' ] th training-test data >>> ******** ********\n')

            # nodeLoc   : 각 node의 3차원상의 위치
            # maxDist   : 각 node가 연결되기 위한 최대의 distance
            # DTNnetwork: 학습 데이터의 DTN 네트워크
            # malArray  : 각 node가 malicious node인지의 여부
            (nodeLoc, maxDist, DTNnetwork, malArray) = ex.makeDTN([20, 25], boardSize=boardSize, maxDist=maxDist, malProb=0.2)
            
            nodes = len(DTNnetwork) # DTN의 노드 개수
            nodeMove = [[0] * nodes for _ in range(nodes)] # nodeMove[A, B]: node A에서 node B로의 이동횟수

            print('<<< node location >>>')
            for i in range(nodes): print('node ' + str(i) + ' : LOC ' + str(nodeLoc[i]))
            print('')

            #print('## Network ##')
            #for j in range(nodes): print(DTNnetwork[j])
            #print('')

            # packet 전달 100회 반복
            malwareInfect = False # 출력용
            for j in range(100):
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
                            distance_kl = math.sqrt(pow(nodeLoc[k][0]-nodeLoc[l][0], 2) + pow(nodeLoc[k][1]-nodeLoc[l][1], 2) + pow(nodeLoc[k][2]-nodeLoc[l][2], 2))

                            # calculate connection probability for the distance
                            connect_prob = conProb(distance_kl/maxDist)
                            # print(distance_kl, maxDist, connect_prob)

                            # connected or not
                            if random.random() < connect_prob: DTNnetwork_[k][l] = DTNnetwork[k][l]
                            else: DTNnetwork_[k][l] = 0

                    # 다음 노드 탐색
                    nextNode = ex.dijkstra_nextnode(thisNode, destNode, DTNnetwork_, malArray[thisNode], False)

                    # 일정 확률로 malicious node 바이러스 전파
                    if malArray[thisNode] == True and random.random() < float(virusProb):
                        malwareInfect = True
                        print('malware infection: ' + str(thisNode) + ' -> ' + str(nextNode))
                        malArray[nextNode] = True

                    # nextNode가 -1이면 startNode -> destNode의 경로가 없다는 의미이므로 건너뛰기
                    if nextNode == -1: break

                    # print(str(j) + ' : ' + str(thisNode) + 'mal:[' + str(malArray[thisNode]) + '] -> ' + str(nextNode) + ' dest: (' + str(destNode) + ')')

                    nodeMove[thisNode][nextNode] += 1 # node A->node B에 대해 nodeMove[A, B] 갱신
                    thisNode = nextNode

                    # nextNode가 destNode와 같으면 destNode에 도착했다는 의미이므로 종료
                    if nextNode == destNode: break

            if malwareInfect == True: print('')
            print('## packet move from A to B ##')
            for j in range(nodes): print(nodeMove[j])

            nodeNo = [] # node 번호 (hop1Noise의 값이 -1, 즉 이웃 node가 없는 경우 제외)

            # [A] 각 node에서 first hop neighbor로 패킷 전송 시의 noise
            # [B] 각 node에서 first hop neighbor로 패킷 전송 시의, 가장 많이 선택되는 node의 선택률
            hop1Noise = []
            hop1Max = []
            for j in range(nodes):
                hop1Noise_j = ex.noise(nodeMove[j])
                if hop1Noise_j >= 0:
                    nodeNo.append(j)
                    hop1Noise.append(hop1Noise_j)
                    hop1Max.append(max(nodeMove[j])/sum(nodeMove[j]))

            # [C] 각 node의 first hop neighbor에서 second hop neighbor로 패킷 전송 시의 noise의 평균값
            # [D] 각 node의 first hop neighbor에서 second hop neighbor로 패킷 전송 시의, 가장 많이 선택되는 node의 선택률의 평균값
            hop2AvgNoise = []
            hop2AvgMax = []
            print('\n## neighbor info ##')
            for j in range(len(nodeNo)): # 각 node에 대하여 (hop1Noise의 값이 -1, 즉 이웃 node가 없는 경우 제외)

                countList.append(count) # dataNo 값 추가

                # hop1Noise의 값이 -1이면 neighbor가 없다는 의미이므로 건너뛰기
                # hop1Noise의 값이 0 이상이면 neighbor가 있다는 의미이고, 각 neighbor에 대해서도 neighbor가 무조건 존재 (해당 node)
                if hop1Noise[j] == -1: continue

                # first hop neighbor를 모두 찾기
                hop1Neighbors = []
                for k in range(nodes):
                    if DTNnetwork[nodeNo[j]][k] > 0: hop1Neighbors.append(k)
                print(str(nodeNo[j]) + ' -> ' + str(hop1Neighbors))

                # 각 first hop neighbor에서 second hop neighbor로 패킷 전송 시의 noise, 가장 많이 선택되는 node의 선택률의 평균값
                avgNoise = 0.0
                avgMax = 0.0
                for k in range(len(hop1Neighbors)):
                    # print(str(j) + ' : ' + str(hop1Neighbors[k]) + ' -> ' + str(ex.noise(nodeMove[hop1Neighbors[k]])))
                    avgNoise += ex.noise(nodeMove[hop1Neighbors[k]])
                    avgMax += max(nodeMove[hop1Neighbors[k]])/sum(nodeMove[hop1Neighbors[k]])
                    
                avgNoise /= len(hop1Neighbors)
                avgMax /= len(hop1Neighbors)

                hop2AvgNoise.append(avgNoise)
                hop2AvgMax.append(avgMax)

            # 각 node의 [A], [B], [C], [D]에 대한 Z 값 (학습 데이터) 생성
            hop1NoiseZ = [] # hop1Noise의 각 원소의 Z값
            hop2AvgNoiseZ = [] # hop2AvgNoise의 각 원소의 Z값

            hop1MaxZ = [] # hop1Max의 각 원소의 Z값
            hop2AvgMaxZ = [] # hop2AvgMax의 각 원소의 Z값
            
            for j in range(len(hop1Noise)):
                hop1NoiseZ.append(ex.Zvalue(hop1Noise, hop1Noise[j]))
                hop1MaxZ.append(ex.Zvalue(hop1Max, hop1Max[j]))
                hop2AvgNoiseZ.append(ex.Zvalue(hop2AvgNoise, hop2AvgNoise[j]))
                hop2AvgMaxZ.append(ex.Zvalue(hop2AvgMax, hop2AvgMax[j]))

            # 다음 배열의 크기는 모두 같아야 함
            # nodeNo, hop1Noise, hop1Max, hop2AvgNoise, hop2AvgMax, hop1NoiseZ, hop1MaxZ, hop2AvgNoiseZ, hop2AvgMaxZ
            assert(len(nodeNo) == len(hop1Noise) and len(hop1Noise) == len(hop1NoiseZ))
            assert(len(hop1Noise) == len(hop1Max) and len(hop1Noise) == len(hop2AvgNoise) and len(hop1Noise) == len(hop2AvgMax))
            assert(len(hop1NoiseZ) == len(hop1MaxZ) and len(hop1NoiseZ) == len(hop2AvgNoiseZ) and len(hop1NoiseZ) == len(hop2AvgMaxZ))

            # [A]와 [B]의 noise data 및 Z 값 출력
            print('\n## noise data (1-hop, 2-hop avg) ##')
            print('dataNo\tnode\tH1N\tH2avgN\tH1NZ\t\tH2avgNZ\t\tH1M\tH2avgM\tH1MZ\t\tH2avgMZ\t\tmal?')
            
            for j in range(len(hop1Noise)):
                toSave_ = str(count) + '\t' + str(nodeNo[j]) + '\t' + str(round(hop1Noise[j], 4)) + '\t' + str(round(hop2AvgNoise[j], 4)) + '\t'
                toSave_ += str(round(hop1NoiseZ[j], 10)) + '\t' + str(round(hop2AvgNoiseZ[j], 10)) + '\t'
                toSave_ += str(round(hop1Max[j], 4)) + '\t' + str(round(hop2AvgMax[j], 4)) + '\t'
                toSave_ += str(round(hop1MaxZ[j], 10)) + '\t' + str(round(hop2AvgMaxZ[j], 10)) + '\t' + str(malArray[nodeNo[j]])
                print(toSave_)

                toSave += toSave_ + '\n'

            # 파일로 저장
            if count == numOfTrain + numOfTest - 1: # 전체 학습 및 테스트 데이터
                f = open('DTN_result.txt', 'w')
                f.write(toSave)
                f.close()
                toSave = ''

            f = open('DTN_result_args.txt', 'w')
            f.write(str(numOfTrain) + ' ' + str(numOfTest) + ' ' + str(virusProb) + '\nboardSize=' + str(boardSize) + ' maxDist=' + str(maxDist))
            f.close()

            # 학습 데이터에 추가
            hop1Noise_ += hop1Noise
            hop1Max_ += hop1Max
            hop2AvgNoise_ += hop2AvgNoise
            hop2AvgMax_ += hop2AvgMax

            hop1NoiseZ_ += hop1NoiseZ
            hop1MaxZ_ += hop1MaxZ
            hop2AvgNoiseZ_ += hop2AvgNoiseZ
            hop2AvgMaxZ_ += hop2AvgMaxZ

            for i in range(len(nodeNo)): isMal_.append(malArray[nodeNo[i]])

            # 카운트 증가
            count += 1
            
        except Exception as e:
            print('\n<---- ERROR: ' + str(e) + ' ---->\n')

    # 결과 반환
    return [[hop1Noise_, hop2AvgNoise_, hop1NoiseZ_, hop2AvgNoiseZ_,
            hop1Max_, hop2AvgMax_, hop1MaxZ_, hop2AvgMaxZ_, isMal_], countList]

def trainAndTest(numOfTrain, numOfTest, virusProb, boardSize, maxDist, deviceName): # 학습 데이터를 읽어서 배열에 저장 및 학습

    # 학습 데이터 (각 x열에 해당하는 8개의 배열, y열에 해당하는 배열)
    # 0 ~ 3 : hop1Noise_, hop2AvgNoise_, hop1NoiseZ_, hop2AvgNoiseZ_,
    # 4 ~ 7 : hop1Max_, hop2AvgMax_, hop1MaxZ_, hop2AvgMaxZ_,
    #   8   : isMal_
    mainData_train = [[], [], [], [], [], [], [], [], []] # 학습 데이터
    mainData_test = [[], [], [], [], [], [], [], [], []] # 테스트 데이터

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

    # 학습 및 테스트 데이터 파일 존재 여부와 무관하게 실행
    try:
        
        # 학습 데이터를 mainFunc을 이용하여 생성
        mF = mainFunc(numOfTrain, numOfTest, virusProb, boardSize, maxDist)
        
        mainData = mF[0] # 학습 및 테스트 데이터의 각 line
        countList = mF[1] # 파일의 각 line에 대한 데이터 번호 (dataNo)
        
        # 숫자 데이터를 float으로 변환
        for i in range(len(mainData)):
            for j in range(8): mainData[j][i] = float(mainData[j][i])

        # numOfTrain_NB, numOfTrain_SVM, numOfTest에 추가
        for i in range(len(mainData[0])):
            dataNoOfThisLine = countList[i] # 데이터 파일의 해당 line의 dataNo 값
            
            if dataInfo[dataNoOfThisLine] == 0: # 0 -> 학습 데이터
                for j in range(9): mainData_train[j].append(mainData[j][i])
            elif dataInfo[dataNoOfThisLine] == 1: # 1 -> 테스트 데이터
                for j in range(9): mainData_test[j].append(mainData[j][i])

    except Exception as e:
        print('\nerror: ' + str(e) + '\n')

    # 학습 데이터, 테스트 데이터의 실제 line 개수 (각 데이터마다 node 개수만큼의 line이 있음)
    lenTrain = len(mainData_train[0])
    lenTest = len(mainData_test[0])

    print('\n################################')
    print(lenTrain, lenTest, virusProb, 'boardSize:', boardSize, 'maxDist:', maxDist)
    print('dataInfo: ' + str(dataInfo))
    print('################################\n')

    # isMal_의 'True', 'False'를 숫자 0(False), 1(True)로 변환
    for i in range(lenTrain):
        if mainData_train[8][i] == True: mainData_train[8][i] = 1
        else: mainData_train[8][i] = 0
    for i in range(lenTest):
        if mainData_test[8][i] == True: mainData_test[8][i] = 1
        else: mainData_test[8][i] = 0

    NN = [tf.keras.layers.Flatten(input_shape=(4,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')]
    op = tf.keras.optimizers.Adam(0.001)

    # 학습 데이터의 입력, 출력 데이터 생성
    # 입력 데이터: hop1NoiseZ_(2), hop2AvgNoiseZ_(3), hop1MaxZ_(6), hop2AvgMaxZ_(7) / 출력 데이터: isMal_(8)
    inputs = []
    outputs = []
    for i in range(lenTrain):
        inputs.append([mainData_train[2][i], mainData_train[3][i], mainData_train[6][i], mainData_train[7][i]])
        outputs.append([mainData_train[8][i]])

    # 테스트 데이터의 입력 데이터 생성
    testInputs = [] # 테스트 입력
    testValOutputs = [] # 테스트에 대한 validation data (실제 결과값)
    for i in range(lenTest):
        testInputs.append([mainData_test[2][i], mainData_test[3][i], mainData_test[6][i], mainData_test[7][i]])
        testValOutputs.append(mainData_test[8][i])

    # 학습 실시
    deepLearning_GPU.deepLearning(NN, op, 'mean_squared_error', inputs, outputs, '200407_DTN', 2500, False, True, deviceName)

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
    correct = TP + FP

    # 테스트 결과 출력
    print('True Positive    : ' + str(TP) + ' / ' + str(round(100 * TP / ALL, 2)) + '%')
    print('True Negative    : ' + str(TN) + ' / ' + str(round(100 * TN / ALL, 2)) + '%')
    print('False Positive   : ' + str(FP) + ' / ' + str(round(100 * FP / ALL, 2)) + '%')
    print('False Negative   : ' + str(FN) + ' / ' + str(round(100 * FN / ALL, 2)) + '%')
    print('')
    print('Real Positive    : ' + str(RealP) + ' / ' + str(round(100 * RealP / ALL, 2)) + '%')
    print('Real Negative    : ' + str(RealN) + ' / ' + str(round(100 * RealN / ALL, 2)) + '%')
    print('')
    print('correct          : ' + str(correct) + ' / ' + str(round(100 * correct / ALL, 2)) + '%')
    print('')

if __name__ == '__main__':
    numOfTrain = int(input('number of train datasets'))
    numOfTest = int(input('number of test datasets'))
    virusProb = float(input('probability of malware virus infection of target node'))
    boardSize = float(input('board Size'))
    maxDist = float(input('maximum distance between 2 nodes to connect them'))
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    
    trainAndTest(numOfTrain, numOfTest, virusProb, boardSize, maxDist, deviceName)
