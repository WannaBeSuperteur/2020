svm = __import__('200325_DTNmodels_svm')
nb = __import__('200325_DTNmodels_nb')
ex = __import__('200325_DTNexample')
import math
import random

def mark(value):
    if value <= -1: return 'D'
    elif value <= 0: return 'C'
    elif value <= 1: return 'B'
    else: return 'A'

# numOfTrain_NB : Naive Bayes 학습 데이터의 개수
# numOfTrain_SVM: SVM 학습 데이터의 개수
# numOfTest     : 테스트 데이터의 개수
# virusProb     : next node가 malware에 감염될 확률
def mainFunc(numOfTrain_NB, numOfTrain_SVM, numOfTest, virusProb):
    count = 0
    toSave = ''

    hop1Noise_ = [] # hop1Noise에 대한 전체 학습 데이터
    hop1Max_ = [] # hop1Max에 대한 전체 학습 데이터
    hop2AvgNoise_ = [] # hop2AvgNoise에 대한 전체 학습 데이터
    hop2AvgMax_ = [] # hop2AvgMax에 대한 전체 학습 데이터
    
    hop1NoiseZ_ = [] # hop1NoiseZ에 대한 전체 학습 데이터
    hop1MaxZ_ = [] # hop1MaxZ에 대한 전체 학습 데이터
    hop2AvgNoiseZ_ = [] # hop2AvgNoiseZ에 대한 전체 학습 데이터
    hop2AvgMaxZ_ = [] # hop2AvgMaxZ에 대한 전체 학습 데이터
    
    hop1NoiseZmark_ = [] # hop1NoiseZmark에 대한 전체 학습 데이터
    hop1MaxZmark_ = [] # hop1MaxZmark에 대한 전체 학습 데이터
    hop2AvgNoiseZmark_ = [] # hop2AvgNoiseZmark에 대한 전체 학습 데이터
    hop2AvgMaxZmark_ = [] # hop2AvgMaxZmark에 대한 전체 학습 데이터
    
    isMal_ = [] # malicious node 여부에 대한 전체 학습 데이터

    # 학습 데이터 생성
    while count < numOfTrain_NB + numOfTrain_SVM + numOfTest: # numOfTrain_NB + numOfTrain_SVM + numOfTest개의 데이터
        try:
            print('\n ******** ******** <<< [ ' + str(count) + ' ] th training-test data >>> ******** ********\n')

            # DTNnetwork: 학습 데이터의 DTN 네트워크
            # malArray  : 각 node가 malicious node인지의 여부
            (DTNnetwork, malArray) = ex.makeDTN([20, 25], 0.1, 0.2)
            
            nodes = len(DTNnetwork) # DTN의 노드 개수
            nodeMove = [[0] * nodes for _ in range(nodes)] # nodeMove[A, B]: node A에서 node B로의 이동횟수

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
                    nextNode = ex.dijkstra_nextnode(thisNode, destNode, DTNnetwork, malArray[thisNode], False)

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

            # 각 node의 [A], [B], [C], [D]에 대한 Z 값 및 학습 데이터 (Z 값의 구간별 마크) 생성
            hop1NoiseZ = [] # hop1Noise의 각 원소의 Z값
            hop1NoiseZmark = [] # hop1Noise의 각 원소의 Z값의 범위에 따른 마크
            hop2AvgNoiseZ = [] # hop2AvgNoise의 각 원소의 Z값
            hop2AvgNoiseZmark = [] # hop2AvgNoise의 각 원소의 Z값의 범위에 따른 마크

            hop1MaxZ = [] # hop1Max의 각 원소의 Z값
            hop1MaxZmark = [] # hop1Max의 각 원소의 Z값의 범위에 따른 마크
            hop2AvgMaxZ = [] # hop2AvgMax의 각 원소의 Z값
            hop2AvgMaxZmark = [] # hop2AvgMax의 각 원소의 Z값의 범위에 따른 마크
            
            for j in range(len(hop1Noise)):
                hop1NoiseZ.append(ex.Zvalue(hop1Noise, hop1Noise[j]))
                hop1MaxZ.append(ex.Zvalue(hop1Max, hop1Max[j]))
                hop2AvgNoiseZ.append(ex.Zvalue(hop2AvgNoise, hop2AvgNoise[j]))
                hop2AvgMaxZ.append(ex.Zvalue(hop2AvgMax, hop2AvgMax[j]))

                # 마크 추가
                hop1NoiseZmark.append(mark(hop1NoiseZ[j]))
                hop1MaxZmark.append(mark(hop1MaxZ[j]))
                hop2AvgNoiseZmark.append(mark(hop2AvgNoiseZ[j]))
                hop2AvgMaxZmark.append(mark(hop2AvgMaxZ[j]))

            # 다음 배열의 크기는 모두 같아야 함
            # nodeNo, hop1Noise, hop1Max, hop2AvgNoise, hop2AvgMax, hop1NoiseZ, hop1MaxZ,
            # hop2AvgNoiseZ, hop2AvgMaxZ, hop1NoiseZmark, hop1MaxZmark, hop2AvgNoiseZmark, hop2AvgMaxZmark
            assert(len(nodeNo) == len(hop1Noise) and len(hop1Noise) == len(hop1NoiseZ) and len(hop1Noise) == len(hop1NoiseZmark))
            assert(len(hop1Noise) == len(hop1Max) and len(hop1Noise) == len(hop2AvgNoise) and len(hop1Noise) == len(hop2AvgMax))
            assert(len(hop1NoiseZ) == len(hop1MaxZ) and len(hop1NoiseZ) == len(hop2AvgNoiseZ) and len(hop1NoiseZ) == len(hop2AvgMaxZ))
            assert(len(hop1NoiseZmark) == len(hop1MaxZmark) and len(hop1NoiseZmark) == len(hop2AvgNoiseZmark) and len(hop1NoiseZmark) == len(hop2AvgMaxZmark))

            # [A]와 [B]의 noise data 및 Z 값 출력
            print('\n## noise data (1-hop, 2-hop avg) ##')
            print('node\tH1N\tH2avgN\tH1NZ\t\tH2avgNZ\t\tH1NM\tH2avgNM\tH1M\tH2avgM\tH1MZ\t\tH2avgMZ\t\tH1MM\tH2avgMM\tmal?')
            
            for j in range(len(hop1Noise)):
                toSave_ = str(nodeNo[j]) + '\t' + str(round(hop1Noise[j], 4)) + '\t' + str(round(hop2AvgNoise[j], 4)) + '\t'
                toSave_ += str(round(hop1NoiseZ[j], 10)) + '\t' + str(round(hop2AvgNoiseZ[j], 10)) + '\t' + str(hop1NoiseZmark[j]) + '\t'
                toSave_ += str(hop2AvgNoiseZmark[j]) + '\t' + str(round(hop1Max[j], 4)) + '\t' + str(round(hop2AvgMax[j], 4)) + '\t'
                toSave_ += str(round(hop1MaxZ[j], 10)) + '\t' + str(round(hop2AvgMaxZ[j], 10)) + '\t' + str(hop1MaxZmark[j]) + '\t'
                toSave_ += str(hop2AvgMaxZmark[j]) + '\t' + str(malArray[nodeNo[j]])
                print(toSave_)

                toSave += toSave_ + '\n'

            # 파일로 저장
            if count == numOfTrain_NB - 1: # Naive Bayes 학습 데이터
                f = open('DTN_result_NB.txt', 'w')
                len0 = len(toSave.split('\n'))-1
                f.write(toSave)
                f.close()
                toSave = ''
            elif count == numOfTrain_NB + numOfTrain_SVM - 1: # SVM 학습 데이터
                f = open('DTN_result_SVM.txt', 'w')
                len1 = len(toSave.split('\n'))-1
                f.write(toSave)
                f.close()
                toSave = ''
            elif count == numOfTrain_NB + numOfTrain_SVM + numOfTest - 1: # 테스트 데이터
                f = open('DTN_result_test.txt', 'w')
                len2 = len(toSave.split('\n'))
                f.write(toSave)
                f.close()
                toSave = ''

            f = open('DTN_result_args.txt', 'w')
            f.write(str(numOfTrain_NB) + ' ' + str(numOfTrain_SVM) + ' ' + str(numOfTest))
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

            hop1NoiseZmark_ += hop1NoiseZmark
            hop1MaxZmark_ += hop1MaxZmark
            hop2AvgNoiseZmark_ += hop2AvgNoiseZmark
            hop2AvgMaxZmark_ += hop2AvgMaxZmark

            for i in range(len(nodeNo)): isMal_.append(malArray[nodeNo[i]])

            # 카운트 증가
            count += 1
            
        except Exception as e:
            print('\n<---- ERROR: ' + str(e) + ' ---->\n')

    # 결과 반환
    return [[hop1Noise_, hop2AvgNoise_, hop1NoiseZ_, hop2AvgNoiseZ_, hop1NoiseZmark_, hop2AvgNoiseZmark_,
            hop1Max_, hop2AvgMax_, hop1MaxZ_, hop2AvgMaxZ_, hop1MaxZmark_, hop2AvgMaxZmark_, isMal_],
            len0, len1, len2]

def trainAndTest(numOfTrain_NB, numOfTrain_SVM, numOfTest, virusProb): # 학습 데이터를 읽어서 배열에 저장 및 학습

    # 학습 데이터 (각 x열에 해당하는 12개의 배열, y열에 해당하는 배열)
    # 0 ~ 5 : hop1Noise_, hop2AvgNoise_, hop1NoiseZ_, hop2AvgNoiseZ_, hop1NoiseZmark_, hop2AvgNoiseZmark_,
    # 6 ~11 : hop1Max_, hop2AvgMax_, hop1MaxZ_, hop2AvgMaxZ_, hop1MaxZmark_, hop2AvgMaxZmark_
    #  12   : isMal_
    mainData_NB = [[], [], [], [], [], [], [], [], [], [], [], [], []] # Naive Bayes용 학습 데이터
    mainData_SVM = [[], [], [], [], [], [], [], [], [], [], [], [], []] # SVM용 학습 데이터
    mainData_test = [[], [], [], [], [], [], [], [], [], [], [], [], []] # 테스트 데이터

    # 학습 데이터 파일이 있으면
    try:
        # Naive Bayes용 학습 데이터 읽기
        fN = open('DTN_result_NB.txt', 'r')
        fNLines = fN.readlines()
        fN.close()

        # Naive Bayes용 학습 데이터를 배열에 추가
        for i in range(len(fNLines)):
            for j in range(13):
                try:
                    mainData_NB[j].append(float(fNLines[i].split('\t')[j+1]))
                except:
                    mainData_NB[j].append(fNLines[i].split('\t')[j+1])

        # SVM용 학습 데이터 읽기
        fS = open('DTN_result_SVM.txt', 'r')
        fSLines = fS.readlines()
        fS.close()

        # SVM용 학습 데이터를 배열에 추가
        for i in range(len(fSLines)):
            for j in range(13):
                try:
                    mainData_SVM[j].append(float(fSLines[i].split('\t')[j+1]))
                except:
                    mainData_SVM[j].append(fSLines[i].split('\t')[j+1])

        # 테스트 데이터 읽기
        fT = open('DTN_result_test.txt', 'r')
        fTLines = fT.readlines()
        fT.close()

        # 테스트 데이터를 배열에 추가
        for i in range(len(fTLines)):
            for j in range(13):
                try:
                    mainData_test[j].append(float(fTLines[i].split('\t')[j+1]))
                except:
                    mainData_test[j].append(fTLines[i].split('\t')[j+1])

        fileExist = True

    # 학습 데이터 파일이 없으면
    except:
        
        # 파일이 없으면 학습 데이터를 mainFunc을 이용하여 직접 생성
        mF = mainFunc(numOfTrain_NB, numOfTrain_SVM, numOfTest, virusProb) # [mainData, # of data in NB, # of data in SVM, # of data in Test]
        mainData = mF[0]

        # 숫자 데이터를 float으로 변환
        for i in range(len(mainData)):
            mainData[0][i] = float(mainData[0][i])
            mainData[1][i] = float(mainData[1][i])
            mainData[2][i] = float(mainData[2][i])
            mainData[3][i] = float(mainData[3][i])

            mainData[6][i] = float(mainData[6][i])
            mainData[7][i] = float(mainData[7][i])
            mainData[8][i] = float(mainData[8][i])
            mainData[9][i] = float(mainData[9][i])

        # numOfTrain_NB, numOfTrain_SVM, numOfTest에 추가
        for i in range(len(mainData[0])):
            if i < mF[1]: # Naive Bayes 학습 데이터
                for j in range(13): mainData_NB[j].append(mainData[j][i])
            elif i < mF[1] + mF[2]: # SVM 학습 데이터
                for j in range(13): mainData_SVM[j].append(mainData[j][i])
            else: # 테스트 데이터
                for j in range(13): mainData_test[j].append(mainData[j][i])

        fileExist = False

    # Naive Bayes 학습데이터, SVM 학습데이터, 테스트 데이터의 실제크기 (각 데이터마다 node 개수만큼의 세부 데이터가 있음)
    lenNB = len(mainData_NB[0])
    lenSVM = len(mainData_SVM[0])
    lenTest = len(mainData_test[0])
    print(lenNB, lenSVM, lenTest)

    # isMal_의 'True', 'False'를 논리값 True, False로 변환
    if fileExist == True:
        for i in range(lenNB):
            if mainData_NB[12][i].split('\n')[0] == 'True': mainData_NB[12][i] = True
            else: mainData_NB[12][i] = False
        for i in range(lenSVM):
            if mainData_SVM[12][i].split('\n')[0] == 'True': mainData_SVM[12][i] = True
            else: mainData_SVM[12][i] = False
        for i in range(lenTest):
            if mainData_test[12][i].split('\n')[0] == 'True': mainData_test[12][i] = True
            else: mainData_test[12][i] = False
    
    # Naive Bayes를 이용하여 SVM에 이용할 학습 데이터의 Malicious or Not을 판정
    # 입력값(NB)         : hop1NoiseZmark_ (4), hop2AvgNoiseZmark_ (5), 출력값: isMal_ (12)
    # 판정할 데이터(SVM) : hop1NoiseZmark_ (4), hop2AvgNoiseZmark_ (5)
    SVM_malOrNot = [] # SVM용 training data에 대해 Naive Bayes의 학습 데이터를 이용, Naive Bayes로 판정한 Malicious or not

    for i in range(lenSVM):
        SVM_malOrNot.append(nb.naiveBayes([mainData_NB[4], mainData_NB[5]], mainData_NB[12], [mainData_SVM[4][i], mainData_SVM[5][i]]))

    # 판정된 Malicious or Not을 SVM에 넣어서 최적의 hyperplane 생성 및 테스트
    # 입력값(SVM)        : hop1MaxZ_ (8), hop2AvgMaxZ_ (9), 출력값: SVM_malOrNot
    # 테스트 데이터(test): hop1MaxZ_ (8), hop2AvgMaxZ_ (9), 출력값: isMal_ (12)

    # 입력값 생성: SVMdata = [[x0, y0], [x1, y1], ...]
    SVMdata = []
    for i in range(lenSVM): SVMdata.append([float(mainData_SVM[8][i]), float(mainData_SVM[9][i])])

    # 테스트 데이터 생성: SVMtestData = [[x0, y0, class0], [x1, y1, class0], ...]
    SVMtestData = []
    for i in range(lenTest): SVMtestData.append([float(mainData_test[8][i]), float(mainData_test[9][i]), mainData_test[12][i]])

    # SVM 학습
    # 그 학습으로 생성된 hyperplane에 대해 testFileName의 파일을 읽어서 그 파일의 데이터를 테스트
    svm.SVM(SVMdata, SVM_malOrNot, SVMtestData, True)

if __name__ == '__main__':
    print("check if 'DTN_result_args.txt' file exists")
    numOfTrain_NB = int(input('number of train dataset for Naive Bayes model'))
    numOfTrain_SVM = int(input('number of train dataset for SVM model'))
    numOfTest = int(input('number of test dataset'))
    virusProb = float(input('probability of malware virus infection of target node'))
    
    trainAndTest(numOfTrain_NB, numOfTrain_SVM, numOfTest, virusProb)
