#!/usr/bin/python2
# source: https://github.com/ztgu/sorting_algorithms_py
# info: [비교 횟수, 이동 횟수]

def swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def bubblesort(array, info):
    n = len(array)-1
    for i in range(0, len(array)):
        for j in range(0, n):
            info[0] += 1 # 비교 횟수 증가
            if array[j] > array[j+1]:
                info[1] += 1 # 이동 횟수 증가
                swap(array, j+1, j)
        n -= 1

    return (info[0], info[1])

if __name__ == "__main__":
    array = [17, 9, 13, 8, 7, -5, 6, 11, 3, 4, 1, 2]
    bubblesort(array, [0, 0])
    print(array)

