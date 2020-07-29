# 1. 6000 families에 대해 전체 prefCost를 계산하는 대신
#    change에 영향이 있는 1 families에 대해서만 계산
#    -> 계산 시간 1/6000으로 단축

# 2. 100 days (500개의 요소)에 대해 전체 account penalty를 계산하는 대신
#    관련 있는 날짜 (바꾸기 전 날짜 및 바꾼 후의 날짜)에 영향이 있는 22 days (20개의 요소)에 대해서만 계산
#    -> 계산 시간 1/25로 단축

# 3. feasible 여부 파악
#    각 날짜별 인원수가 저장된 배열(occupancy array)을 이용한다.

# 1의 계산 시간이 600초, 2의 계산 시간이 60초라고 하면
# 기존 660.0초 -> 개선 0.1초 + 2.4초 = 2.5초 (약 1/260)

# 4. day (X-5) ~ day (X+5) 부분을 추출하여 account penalty 계산
# penalty[i] = [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[min(i+j, 100)]))/(j*j)] * [(ocA[i] - 125)/400] (j=1,2,3,4,5)
#              에서 day i에 대한 penalty를 구하려면 ocA[i], ocA[i+1], ..., ocA[i+5]에 대한 정보가 필요
#
# penalty[i] = [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[i+1]))] * [(ocA[i] - 125)/400] +
#              [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[i+2]))/4] * [(ocA[i] - 125)/400] +
#              [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[i+3]))/9] * [(ocA[i] - 125)/400] +
#              [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[i+4]))/16] * [(ocA[i] - 125)/400] +
#              [pow(ocA[i], 0.5 + 0.02*abs(ocA[i]-ocA[i+5]))/25] * [(ocA[i] - 125)/400]
# 따라서 ocA[X]와 관련된 모든 penalty의 합은
# [pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+1])) + ... + pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+5])) / 25] * [(ocA[X] - 125)/400] +
# [pow(ocA[X-1], 0.5 + 0.02*abs(ocA[X-1]-ocA[X]))] * [(ocA[X-1] - 125)/400] +
# [pow(ocA[X-2], 0.5 + 0.02*abs(ocA[X-2]-ocA[X])) / 4] * [(ocA[X-2] - 125)/400] +
# [pow(ocA[X-3], 0.5 + 0.02*abs(ocA[X-3]-ocA[X])) / 9] * [(ocA[X-3] - 125)/400] +
# [pow(ocA[X-4], 0.5 + 0.02*abs(ocA[X-4]-ocA[X])) / 16] * [(ocA[X-4] - 125)/400] +
# [pow(ocA[X-5], 0.5 + 0.02*abs(ocA[X-5]-ocA[X])) / 25] * [(ocA[X-5] - 125)/400]

# ocA의 변화량에 따라 account penalty가 달라지므로 변화량을 가능한 한 줄이는 것이 중요

# 매 턴마다 다음과 같이 변경
# 00 초기화한다.
# 01 while True:
# 02     현재 상태(=기존 옵션)에 대한 score를 계산하여 출력한다. (feasible하지 않으면 무한대)
# 03     For 모든 families
# 04         p0 = 해당 family에 대한 prefCost를 계산한다.
# 05         각 option에 따른 score의 변화량을 저장한 배열을 초기화한다.
# 06         For 모든 options (choice_0, choice_1, ..., choice_9)
# 07             변경전 day의 인원수(ocA[before]) - 해당 family 인원수가 125 미만이면 제외
# 08             변경후 day의 인원수(ocA[after]) + 해당 family 인원수가 300 초과이면 제외
# 09             B0 = 변경전 day가 before일 때 day (before-5) ~ day (before+5) 부분을 추출하여 account penalty를 계산한다.
# 10             A0 = 변경후 day가 after일 때 day (after-5) ~ day (after+5) 부분을 추출하여 account penalty를 계산한다.
# 11             ocA를 복사한 배열을 ocA_copy라 하면 ocA_copy[before] -= people; ocA_copy[after] += people;
# 12             B1 = 변경전 day가 before일 때 day (before-5) ~ day (before+5) 부분을 추출하여 account penalty를 계산한다.
# 13             A1 = 변경후 day가 after일 때 day (after-5) ~ day (after+5) 부분을 추출하여 account penalty를 계산한다.
# 14             p1 = 해당 family에 대한 변경된 prefCost를 계산한다.
# 15             해당 family에 대한 account penalty의 변화량 (B1+A1)-(B0+A0), prefCost의 변화량 (p1-p0)을 이용하여 score의 변화량 (B1+A1+p1)-(B0+A0+p0)을 계산한다.
# 16         if 기존 option에 대한 score보다 낮은 score인 option이 있으면 (즉 score의 변화량 < 0)
# 17             score(의 변화량)가 가장 작은 option을 찾아서 실행한다.
# 18             occupancy array를 업데이트한다. (ocA[before] -= people; ocA[after] += people;)
# 19     더 이상 최적화가 되지 않으면 종료


##### 버그 수정사항 #####
# after가 100일 때 account penalty 계산
# before와 after가 같을 때 (73) account penalty 계산 (실제로 0이지만 계산시 -로 나올때가 있음, 완료)
# 거의 수렴했을 때 ocA 출력 테스트

import helper as h
import random
import math

INFINITE = 2147483647
FAMILIES = 6000
DAYS = 100
OPTIONS = 10

# 11 days 정보에 의한 account penalty
# array = [ocA[X-5], ocA[X-4], ..., ocA[X], ocA[X+1], ..., ocA[X+5]]

# ocA[X]와 관련된 모든 penalty의 합은
# [pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+1])) + ... + pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+5])) / 25] * [(ocA[X] - 125)/400] +
# [pow(ocA[X-1], 0.5 + 0.02*abs(ocA[X-1]-ocA[X]))] * [(ocA[X-1] - 125)/400] +
# [pow(ocA[X-2], 0.5 + 0.02*abs(ocA[X-2]-ocA[X])) / 4] * [(ocA[X-2] - 125)/400] +
# [pow(ocA[X-3], 0.5 + 0.02*abs(ocA[X-3]-ocA[X])) / 9] * [(ocA[X-3] - 125)/400] +
# [pow(ocA[X-4], 0.5 + 0.02*abs(ocA[X-4]-ocA[X])) / 16] * [(ocA[X-4] - 125)/400] +
# [pow(ocA[X-5], 0.5 + 0.02*abs(ocA[X-5]-ocA[X])) / 25] * [(ocA[X-5] - 125)/400]

# X=100이면
# pow(ocA[100], 0.5 + 0.02*abs(ocA[100]-ocA[100])) = A 이면 (A + A/4 + A/9 + A/16 + A/25) * [(ocA[100]-125)/400]
# + pow(ocA[99], 0.5 + 0.02*abs(ocA[99]-ocA[100])) = B 이면 (B + B/4 + B/9 + B/16 + B/25) * [(ocA[99]-125)/400]
# + pow(ocA[98], 0.5 + 0.02*abs(ocA[98]-ocA[100])) = C 이면 (C/4 + C/9 + C/16 + C/25) * [(ocA[98]-125)/400]
# + pow(ocA[97], 0.5 + 0.02*abs(ocA[97]-ocA[100])) = D 이면 (D/9 + D/16 + D/25) * [(ocA[97]-125)/400]
# + pow(ocA[96], 0.5 + 0.02*abs(ocA[96]-ocA[100])) = E 이면 (E/16 + E/25) * [(ocA[96]-125)/400]
# + pow(ocA[95], 0.5 + 0.02*abs(ocA[95]-ocA[100])) = F 이면 (F/25) * [(ocA[95]-125)/400]

# X=96~99이면
# pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+1])) = A[1]
# pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+2])) = A[2]
# pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+3])) = A[3]
# pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+4])) = A[4]
# pow(ocA[X], 0.5 + 0.02*abs(ocA[X]-ocA[X+5])) = A[5] 이면
#
# [X=99] (A[1] + A[1]/4 + A[1]/9 + A[1]/16 + A[1]/25) * [(ocA[99]-125)/400]
# [X=98] (A[1] + A[2]/4 + A[2]/9 + A[2]/16 + A[2]/25) * [(ocA[98]-125)/400]
# [X=97] (A[1] + A[2]/4 + A[3]/9 + A[3]/16 + A[3]/25) * [(ocA[97]-125)/400]
# [X=96] (A[1] + A[2]/4 + A[3]/9 + A[4]/16 + A[4]/25) * [(ocA[96]-125)/400]
#
# pow(ocA[X-1], 0.5 + 0.02*abs(ocA[X-1]-ocA[X])) = B 이면 B * [(ocA[X-1]-125)/400]
# pow(ocA[X-2], 0.5 + 0.02*abs(ocA[X-2]-ocA[X])) = C 이면 C/4 * [(ocA[X-2]-125)/400]
# pow(ocA[X-3], 0.5 + 0.02*abs(ocA[X-3]-ocA[X])) = D 이면 D/9 * [(ocA[X-3]-125)/400]
# pow(ocA[X-4], 0.5 + 0.02*abs(ocA[X-4]-ocA[X])) = E 이면 E/16 * [(ocA[X-4]-125)/400]
# pow(ocA[X-5], 0.5 + 0.02*abs(ocA[X-5]-ocA[X])) = F 이면 F/25 * [(ocA[X-5]-125)/400]
#
# -> X=100이 아니면 100을 초과하는 index의 경우는 ocA[100]을 적용하면 됨
def accountPenalty(array):
    result = 0

    result += (pow(array[5], 0.5 + 0.02*abs(array[5]-array[6]))) * ((array[5]-125)/400)
    result += (pow(array[5], 0.5 + 0.02*abs(array[5]-array[7])) / 4) * ((array[5]-125)/400)
    result += (pow(array[5], 0.5 + 0.02*abs(array[5]-array[8])) / 9) * ((array[5]-125)/400)
    result += (pow(array[5], 0.5 + 0.02*abs(array[5]-array[9])) / 16) * ((array[5]-125)/400)
    result += (pow(array[5], 0.5 + 0.02*abs(array[5]-array[10])) / 25) * ((array[5]-125)/400)

    if array[4] != None: result += (pow(array[4], 0.5 + 0.02*abs(array[4]-array[5]))) * ((array[4]-125)/400)
    if array[3] != None: result += (pow(array[3], 0.5 + 0.02*abs(array[3]-array[5])) / 4) * ((array[3]-125)/400)
    if array[2] != None: result += (pow(array[2], 0.5 + 0.02*abs(array[2]-array[5])) / 9) * ((array[2]-125)/400)
    if array[1] != None: result += (pow(array[1], 0.5 + 0.02*abs(array[1]-array[5])) / 16) * ((array[1]-125)/400)
    if array[0] != None: result += (pow(array[0], 0.5 + 0.02*abs(array[0]-array[5])) / 25) * ((array[0]-125)/400)
    
    return result

# X=100인 경우의 관련된 모든 penalty의 합
# pow(ocA[100], 0.5 + 0.02*abs(ocA[100]-ocA[100])) = A 이면 (A + A/4 + A/9 + A/16 + A/25) * [(ocA[100]-125)/400]
# + pow(ocA[99], 0.5 + 0.02*abs(ocA[99]-ocA[100])) = B 이면 (B + B/4 + B/9 + B/16 + B/25) * [(ocA[99]-125)/400]
# + pow(ocA[98], 0.5 + 0.02*abs(ocA[98]-ocA[100])) = C 이면 (C/4 + C/9 + C/16 + C/25) * [(ocA[98]-125)/400]
# + pow(ocA[97], 0.5 + 0.02*abs(ocA[97]-ocA[100])) = D 이면 (D/9 + D/16 + D/25) * [(ocA[97]-125)/400]
# + pow(ocA[96], 0.5 + 0.02*abs(ocA[96]-ocA[100])) = E 이면 (E/16 + E/25) * [(ocA[96]-125)/400]
# + pow(ocA[95], 0.5 + 0.02*abs(ocA[95]-ocA[100])) = F 이면 (F/25) * [(ocA[95]-125)/400]

# 100을 초과하는 index의 경우는 ocA[100]을 적용
def accountPenaltyFor100(array):
    A = pow(array[5], 0.5) * ((array[5]-125)/400) # 0.02*abs(ocA[100]-ocA[100]) = 0
    B = pow(array[4], 0.5 + 0.02*abs(array[4]-array[5])) * ((array[4]-125)/400)
    C = pow(array[3], 0.5 + 0.02*abs(array[3]-array[5])) * ((array[3]-125)/400)
    D = pow(array[2], 0.5 + 0.02*abs(array[2]-array[5])) * ((array[2]-125)/400)
    E = pow(array[1], 0.5 + 0.02*abs(array[1]-array[5])) * ((array[1]-125)/400)
    F = pow(array[0], 0.5 + 0.02*abs(array[0]-array[5])) * ((array[0]-125)/400)

    sumA = A + A/4 + A/9 + A/16 + A/25
    sumB = B + B/4 + B/9 + B/16 + B/25
    sumC = C/4 + C/9 + C/16 + C/25
    sumD = D/9 + D/16 + D/25
    sumE = E/16 + E/25
    sumF = F/25

    return sumA + sumB + sumC + sumD + sumE + sumF

# 해당 family에 대한 prefCost
# famData_i = famData[i], subData_i = subData[i]
def prefCostForThisFamily(famData_i, subData_i):

    result = 0
    date = subData_i[1]
    members = famData_i[11]

    if famData_i[1] == date: # choice_0
        result += 0
    elif famData_i[2] == date: # choice_1
        result += 50
    elif famData_i[3] == date: # choice_2
        result += (50 + 9 * members)
    elif famData_i[4] == date: # choice_3
        result += (100 + 9 * members)
    elif famData_i[5] == date: # choice_4
        result += (200 + 9 * members)
    elif famData_i[6] == date: # choice_5
        result += (200 + 18 * members)
    elif famData_i[7] == date: # choice_6
        result += (300 + 18 * members)
    elif famData_i[8] == date: # choice_7
        result += (300 + 36 * members)
    elif famData_i[9] == date: # choice_8
        result += (400 + 36 * members)
    elif famData_i[10] == date: # choice_9
        result += (500 + 235 * members)
    else: # otherwise
        result += (500 + 434 * members)
    
    return result

def minNone(array):
    newArray = []
    index = 0
    minVal = INFINITE
    
    for i in range(len(array)):
        if array[i] != None:
            if array[i] < minVal:
                minVal = array[i]
                index = i

    return [minVal, index]

def roundPrint(array, n):
    newArray = []
    for i in range(len(array)):
        if array[i] == None: newArray.append(None)
        else: newArray.append(round(array[i], n))
    print(newArray)

# FAMILIES와 OPTIONS를 이용하여 최선의 옵션을 선택
# fam은 family ID, opt는 0, 1, 2, ..., or 9
#
# 단일 변경 옵션 0, 1, ..., N:
#     [
#      [[fam0, opt0]], # option 0은 1개를 변경
#      [[fam1, opt1]], # option 1은 1개를 변경
#      ...,
#      [[famN, optN]]  # option N은 1개를 변경
#     ]
#
# 복수 변경 옵션 0, 1, ..., N:
#     [
#      [[fam00, opt00], [fam01, opt01], ..., [fam0A, opt0A]], # option 0은 (A+1)개를 변경
#      [[fam10, opt10], [fam11, opt11], ..., [fam1B, opt1B]], # option 1은 (B+1)개를 변경
#      ...,
#      [[famN0, optN0], [famN1, optN1], ..., [famNC, optNC]]  # option N은 (C+1)개를 변경
#     ]
#
# ci(기존 count), i는 count용 변수, prevScore는 직전 점수
def findBestOption(subData, famData, options, ocA, ci, i, prevScore):

    # 04 각 option에 따른 score의 변화량을 저장한 배열을 초기화한다.
    scoreChange = []
    for j in range(len(options)): scoreChange.append(None)

    befores = [] # value of 'before' for each option
    afters = [] # value of 'after' for each option

    if ci == 0 and i > 2300 and i < 2350: print('k\tA1s\tB1s\tp1s\tall(1)\tA0s\tB0s\tp0s\tall(0)\tdif') # temp
            
    # 05 For 모든 options (0, 1, 2, ..., and 9)
    for j in range(len(options)):

        thisOption = options[j] # 변경사항: [[famK0, optK0], [famK1, optK1], ..., [famKA, optKA]]

        # 06 p0 = 해당 family에 대한 prefCost를 계산한다.
        #    p0 : 해당 family에 대한 date 변경 전 prefCost
        p0 = prefCostForThisFamily(famData[i], subData[i])

        # 07 ocA를 복사한 배열을 만든다.
        ocA_copy = []
        for k in range(len(ocA)): ocA_copy.append(ocA[k])

        # 각 option에 대하여 (famK0, famK1, ..., famKA에 대하여 변경)
        for k in range(len(thisOption)):
            famID = thisOption[k][0] # option의 변경사항 k에 대한 family ID
            toChange = thisOption[k][1] # option의 변경사항 k에 대한 change to this option (0, 1, ..., or 9)

            before = subData[famID][1] # 변경 전 date (option)
            after = famData[famID][toChange+1] # 변경 후 date (option)
            members = famData[famID][11]

            befores.append(before)
            afters.append(after)

            # before와 after가 서로 같으면 변화가 없는 것이므로 건너뛰기
            if before == after: continue
                    
            # 08 변경전 day의 인원수(ocA[before]) - 해당 family 인원수가 125 미만이면 제외
            # 09 변경후 day의 인원수(ocA[after]) + 해당 family 인원수가 300 초과이면 제외
            if ocA[before] - members < 125 or ocA[after] + members > 300: continue

            # 10 B0 = 변경전 day가 before일 때 day (before-5) ~ day (before+5) 부분을 추출하여 account penalty를 계산한다.
            #    B0 : 변경 전 date의 ocA 변경 전 account penalty
            beforeOCA = [None]*max(6-before, 0) + ocA[max(before-5, 1):min(before+6, 101)] + [ocA[100]]*max(0, before-95)

            if k == 0: # k=0일 때만 B0에 반영
                if before == 100: B0 = accountPenaltyFor100(beforeOCA)
                else: B0 = accountPenalty(beforeOCA)

            # 11 A0 = 변경후 day가 after일 때 day (after-5) ~ day (after+5) 부분을 추출하여 account penalty를 계산한다.
            #    A0 : 변경 후 date의 ocA 변경 전 account penalty
            afterOCA = [None]*max(6-after, 0) + ocA[max(after-5, 1):min(after+6, 101)] + [ocA[100]]*max(0, after-95)

            if k == 0: # k=0일 때만 A0에 반영
                if after == 100: A0 = accountPenaltyFor100(afterOCA)
                else: A0 = accountPenalty(afterOCA)

            # 12 [IMPORTANT] ocA 배열 갱신: ocA_copy[before] -= people; ocA_copy[after] += people;
            for l in range(len(ocA_copy)):
                if l == before and l != after: ocA_copy[l] -= members
                elif l != before and l == after: ocA_copy[l] += members
                
            # 13 B1 = 변경전 day가 before일 때 day (before-5) ~ day (before+5) 부분을 추출하여 account penalty를 계산한다.
            #    B1 : 변경 전 date의 ocA 변경 후 account penalty
            beforeOCA_copy = [None]*max(6-before, 0) + ocA_copy[max(before-5, 1):min(before+6, 101)] + [ocA_copy[100]]*max(0, before-95)

            if k == len(thisOption)-1: # k가 마지막일 때만 B1에 반영
                if before == 100: B1 = accountPenaltyFor100(beforeOCA_copy)
                else: B1 = accountPenalty(beforeOCA_copy)

            # 14 A1 = 변경후 day가 after일 때 day (after-5) ~ day (after+5) 부분을 추출하여 account penalty를 계산한다.
            #    A1 : 변경 후 date의 ocA 변경 후 account penalty
            afterOCA_copy = [None]*max(6-after, 0) + ocA_copy[max(after-5, 1):min(after+6, 101)] + [ocA_copy[100]]*max(0, after-95)

            if k == len(thisOption)-1: # k가 마지막일 때만 A1에 반영
                if after == 100: A1 = accountPenaltyFor100(afterOCA_copy)
                else: A1 = accountPenalty(afterOCA_copy)

            # 15 p1 = 해당 family에 대한 변경된 prefCost를 계산한다.
            #    p1 : 해당 family에 대한 date 변경 후 prefCost
            if k == len(thisOption)-1: # k가 마지막일 때만 p1에 반영
                p1 = prefCostForThisFamily(famData[famID], [famID, after])

        # 16 해당 option에 대한 account penalty의 변화량 [(A1s+B1s)-(A0s+B0s)],
        # prefCost의 변화량 (p1s-p0s)을 이용하여 score의 변화량 (B1s+A1s+p1s)-(B0s+A0s+p0s)을 계산한다.
        try:
            scoreChange[j] = (A1 + B1 + p1) - (A0 + B0 + p0)
        except:
            scoreChange[j] = None

    # scoreChange가 모두 None이면 건너뛰기
    allNone = []
    for j in range(len(options)): allNone.append(None)
    if scoreChange == allNone: return prevScore

    # 17 기존 option에 대한 score보다 낮은 score인 option이 있으면 (즉 score의 변화량 < 0)
    if minNone(scoreChange)[0] < 0:

        if ci == 0 and i > 2300 and i < 2350: roundPrint(scoreChange, 0) # temp

        # 18 score(의 변화량)가 가장 작은 (음수 중에서는 절댓값이 가장 큰) option을 찾아서 실행한다.
        minBefore = 0 # 가장 작은 option의 before 값
        minAfter = 0 # 가장 작은 option의 after 값
        minFamIDs = [] # 가장 작은 option의 family ID의 집합
        minChoiceSC = 0 # 가장 작은 option의 score의 변화량
        minChoices = [] # 가장 작은 option의 option 번호(0~9)의 집합

        for k in range(len(scoreChange)):
            if scoreChange[k] == None: continue
            if scoreChange[k] < minChoiceSC:
                minChoiceSC = scoreChange[k]

                # 해당 option의 family ID 및 option 번호 추출
                thisOption = options[k]
                minFamIDs = []
                minChoices = []
                for l in range(len(thisOption)):
                    minFamIDs.append(thisOption[l][0])
                    minChoices.append(thisOption[l][1])

        # subData 변경
        for k in range(len(minFamIDs)):
            subData[minFamIDs[k]][1] = famData[minFamIDs[k]][minChoices[k]+1]

            # 19 occupancy array를 업데이트한다. (ocA[before of minChoice] -= people; ocA[after of minChoice] += people;)
            ocA[befores[minChoices[k]]] -= members
            ocA[afters[minChoices[k]]] += members

        score = h.getScoreFeasible(famData, subData, ocA)

        if ci == 0 and i > 2300 and i < 2350: print(score) # temp

        return score

    # 가장 작은 변화량이 0 초과이면 변화가 없으므로 이전 값을 반환
    return prevScore

if __name__ == '__main__':

    # 00 초기화한다.
    # read family data
    familyData = open('family_data.csv', 'r')
    data0 = familyData.readlines()
    familyData.close()

    famData = h.makeTable(data0, 1)

    # read sample submission data
    sampleSub = open('sample_submission.csv', 'r')
    data1 = sampleSub.readlines()
    sampleSub.close()

    subData = h.makeTable(data1, 1)
    lastScore = INFINITE
    count = -1

    lastStageScore = 0
    
    # 01 while True:
    while True:
        count += 1
        
        # 02 현재 상태에 대한 score를 계산하여 출력한다. (feasible하지 않으면 무한대)
        ocA = h.getOccupancy(famData, subData) # occupancy array
        score = h.getScoreFeasible(famData, subData, ocA)

        # 지난 단계의 score와 같으면 더 이상 업데이트가 되지 않으므로 랜덤하게 5개 변경 (미구현)
        random5 = False
        if lastStageScore == score: random5 = True

        # 지난 단계(while문 loop)의 score를 업데이트
        lastStageScore = score

        # 03 For 모든 families
        for i in range(FAMILIES):

            if (count == 0 and i % 50 == 0) or (count > 0 and i % 1000 == 0): print(str(count) + ' ' + str(i) + ' / score: ' + str(score))

            options = [[[i, 0]], [[i, 1]], [[i, 2]], [[i, 3]], [[i, 4]], [[i, 5]], [[i, 6]], [[i, 7]], [[i, 8]], [[i, 9]]]
            score = findBestOption(subData, famData, options, ocA, count, i, score)

        # 19 더 이상 최적화가 되지 않으면 종료
        if score < 114000: break
