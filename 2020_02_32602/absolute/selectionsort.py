#!/usr/bin/python2
# source: https://github.com/ztgu/sorting_algorithms_py
# info: [비교 횟수, 이동 횟수]

def swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def selectionsort(array, info):
    global compares
    global moves
    
    for i in range(0, len(array)):
        Ismallest = i
        for j in range(i+1, len(array)):
            info[0] += 1 # 비교 횟수 증가
            if array[j] < array[Ismallest]:
                Ismallest = j
        info[1] += 1 # 이동 횟수 증가
        swap(array, i, Ismallest)

    return (info[0], info[1])

if __name__ == "__main__":
    array = [17, 9, 13, 8, 7, -5, 6, 11, 3, 4, 1, 2]
    selectionsort(array, [0, 0])
    print(array)
