import math
import random
import numpy as np

# malicious node가 아니면 Dijkstra 알고리즘을 이용하여 destination node로 이동하기 위한 다음 node 탐색
# (malicious node이면 이웃한 노드 중 랜덤하게 다음 node로 지정)
# nodeNo : 현재 node의 번호
# destNo : destination node의 번호
# graph  : 인접행렬로 나타낸 그래프 (값: 연결되어 있으면 distance의 값, 그렇지 않으면 0)
# isMal  : 현재 node가 malicious node이면 true, 그렇지 않으면 false
# isPrint: 관련 정보 출력 여부
def dijkstra_nextnode(nodeNo, destNo, graph, isMal, isPrint):
    
    nodes = len(graph) # 전체 node의 개수
    assert(len(graph) == len(graph[0]) and nodeNo < nodes and destNo < nodes)

    # malicious node이면 다음 node를 현재 node와 이웃한 node들 중에서 랜덤하게 선택하기
    if isMal:
        while True:
            
            # 이웃한 node가 없으면 -1을 반환
            if sum(graph[nodeNo]) == 0: return -1

            # 이웃한 node 찾기
            a = random.randint(0, nodes-1)
            if graph[nodeNo][a] > 0: return a

    # malicious node가 아니면 다음 node를 Dijkstra algorithm에 따라 최적으로 결정
    else:
        queue = [(nodeNo, 0)] # 현재 탐색 중인 node를 저장한 queue (구조: [(node0, dist0), ..., (nodeK, distK)]
        finished = [] # 탐색이 완료된 node를 저장한 queue
        previousNode = [-1] * nodes # dijkstra 알고리즘으로 만들어진 최단경로 트리에서 해당 node의 부모 node

        # create distance array
        distanceArray = [9999] * nodes
        distanceArray[nodeNo] = 0
    
        # find next node using dijkstra algorithm
        while len(queue) > 0:
            if isPrint: print('queue (nodeNo, distance):' + str(queue))

            thisNo = queue.pop(0)[0] # 현재 queue에서 뽑아낸 node의 번호 (탐색 중인 node)
            if isPrint: print('searching node: ' + str(thisNo))

            # 각 node에 대하여
            for i in range(nodes):
                if graph[thisNo][i] > 0: # 탐색 중인 node가 node i와 연결되어 있으면

                    # distance(i) = min(distance(i), distance(node)+distance(node, i))
                    # distance(i)를 업데이트해야 하는 경우
                    if distanceArray[i] > distanceArray[thisNo] + graph[thisNo][i]:
                        distanceArray[i] = distanceArray[thisNo] + graph[thisNo][i]
                        previousNode[i] = thisNo

                    # check if node i is already in the queue
                    alreadyInTheQueue = False
                    for j in range(len(queue)):
                        if queue[j][0] == i:
                            alreadyInTheQueue = True
                            break

                    # add node i to the queue if not finished and not alreadyInTheQueue
                    if not i in finished and not alreadyInTheQueue: queue.append((i, distanceArray[i]))

            # sort the queue
            for i in range(len(queue)): queue[i] = (queue[i][0], distanceArray[queue[i][0]]) # update distance in queue
            queue.sort(key=lambda element:element[1]) # sort the queue

            # append node thisNo to finished array
            finished.append(thisNo)

        if isPrint: print('\nnode structure (previous node for each node):')
        if isPrint: print(previousNode)
            
        # destNo 노드로 가기 위한 nodeNo의 다음 node 찾기
        nextNode = destNo
        while previousNode[nextNode] != nodeNo:
            nextNode = previousNode[nextNode]
            if nextNode == -1: return -1
        return nextNode

# graph 인접행렬과 isMalForEachNode(각 node가 malicious한지의 여부)로 구성된 DTN을 반환
# rangeOfNodes: node의 개수 범위 [minNodes, maxNodes] including both end points
# connectProb : 임의의 node A와 node B가 연결되어 있을 확률
# malProb     : 임의의 node가 악성 node일 확률
def makeDTN(rangeOfNodes, connectProb, malProb):
    nodes = random.randint(rangeOfNodes[0], rangeOfNodes[1])
    graph = [[0] * nodes for _ in range(nodes)]

    # connect each node
    for i in range(nodes):
        for j in range(nodes):
            if i != j and random.random() < connectProb: # connectProb의 확률로 연결
                rand = random.randint(1, 9)
                graph[i][j] = rand
                graph[j][i] = rand

    # malicious node
    isMalForEachNode = [False] * nodes # 각 node가 malicious node인지 설정
    for i in range(nodes):
        if random.random() < malProb: # 각 node에 대해 malProb의 확률로 malicious node로 지정
            isMalForEachNode[i] = True

    return (graph, isMalForEachNode)

# noise
# (noise) = (entropy) = -Sum(pi * log2(pi))
# pArray: noise를 계산할 각 pi를 저장한 배열로, 합이 1이 아닌 경우 각 원소를 pArray 배열의 값들의 합으로 나눔
def noise(pArray):
    pArraySum = sum(pArray)
    if pArraySum == 0: return -1 # error

    # sum of pArray[i] <- 1.0
    for i in range(len(pArray)): pArray[i] /= pArraySum

    # 0 초과의 각 원소에 대해 pi * log2(pi) 의 값을 구하고 그것을 합산
    result = 0
    for i in range(len(pArray)):
        if pArray[i] > 0: result -= pArray[i] * math.log(pArray[i], 2)
    return result

# Z value
# Z = (X-m)/(stddev)
# array: 평균과 표준편차를 구할 데이터 자료
# X    : Z value를 구할 값
def Zvalue(array, X):
    avg = np.mean(array)
    stddev = np.std(array)
    return (X - avg) / stddev

if __name__ == '__main__':
    # https://www.researchgate.net/figure/Illustration-of-Dijkstras-algorithm-The-starting-or-source-vertex-s-is-the-leftmost_fig6_4283137
    nodeNo = 0
    destNo = 4
    graph = [[0, 10, 0, 5, 0],
             [0, 0, 1, 2, 0],
             [0, 0, 0, 0, 4],
             [0, 3, 9, 0, 2],
             [7, 0, 6, 0, 0]]
    print(dijkstra_nextnode(nodeNo, destNo, graph, False, True))

    # http://users.cecs.anu.edu.au/~Alistair.Rendell/Teaching/apac_comp3600/module4/single_source_shortest_paths.xhtml
    nodeNo = 0
    destNo = 8
    graph = [[0, 5, 0, 3, 0, 0, 0, 0, 0],
             [0, 0, 8, 0, 2, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 5, 0, 2, 0, 0],
             [0, 0, 0, 0, 0, 3, 0, 1, 0],
             [0, 0, 2, 0, 0, 0, 0, 0, 2],
             [0, 0, 0, 0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 5],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    print(dijkstra_nextnode(nodeNo, destNo, graph, False, True))

    # https://www.leda-tutorial.org/en/unofficial/ch05s03s03.html
    nodeNo = 0
    destNo = 15
    graph = [[0, 0, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 2, 0, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 0],
             [3, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 3, 0, 3, 0, 1, 0, 1, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0],
             [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0]]
    print(dijkstra_nextnode(nodeNo, destNo, graph, False, True))

    # test data with error
    nodeNo = 3
    destNo = 4
    graph = [[0, 0, 8, 0, 0, 6, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 7, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 7, 0, 0, 0, 9, 0, 0],
            [0, 0, 0, 0, 7, 0, 8, 4, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
            [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6],
            [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 7, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 1, 0, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 0, 0, 0, 0, 2, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 6, 3, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0]]
    print(dijkstra_nextnode(nodeNo, destNo, graph, False, True))

    (g, m) = makeDTN([15, 20], 0.25, 0.05)
    print('')
    print('graph:')
    print(g)
    print('malicious?:')
    print(m)

    print('')
    print('noise of [1, 1]: ' + str(noise([1, 1])))
    print('noise of [1, 1, 2, 4]: ' + str(noise([1, 1, 2, 4])))
    print('noise of [0, 1, 1, 0, 2, 4, 0, 0]: ' + str(noise([0, 1, 1, 0, 2, 4, 0, 0])))
    print('Z value of 3 in [0, 2, 5]: ' + str(Zvalue([0, 2, 5], 3)))
