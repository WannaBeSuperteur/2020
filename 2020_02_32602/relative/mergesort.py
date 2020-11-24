#!/usr/bin/python2
# source: https://github.com/ztgu/sorting_algorithms_py
# info: [비교 횟수, 이동 횟수]

import sys # added
sys.setrecursionlimit(9999)

def _mergesort(array, start, end, info):
    
    """ Recursive mergesort function """
    # split
    mid = (start + end)/2
    info[0] += 1 # 비교 횟수 증가
    if start < end:
        _mergesort(array, start, mid, info)
        _mergesort(array, mid+1, end, info)
    elif start == end: return

    # merging Left and Right array
    # originally L = start; R = mid+1
    L = int(start); R = int(mid)+1
    tmp_array = []

    info[0] += 1 # 비교 횟수 증가 (while)
    while ( L <= mid and R <= end):
        info[0] += 1 # 비교 횟수 증가 (if)
        if (array[L] < array[R]):
            info[1] += 1 # 이동 횟수 증가
            tmp_array.append(array[L])
            L += 1
        else:
            info[1] += 1 # 이동 횟수 증가
            tmp_array.append(array[R])
            R += 1

        info[0] += 1 # 비교 횟수 증가 (while)

    # append remaining list, Left array
    info[0] += 1 # 비교 횟수 증가
    if L <= mid:
        info[1] += 1 # 이동 횟수 증가
        tmp_array += array[L:]
    else:
        info[1] += 1 # 이동 횟수 증가
        tmp_array += array[R:]

    # tmp_array to array
    i = 0;
    info[0] += 1 # 비교 횟수 증가 (while)
    while (start <= end):
        info[1] += 1 # 이동 횟수 증가

        # originally array[start] = tmp_array[i]
        array[int(start)] = tmp_array[i]
        start += 1; i += 1;

        info[0] += 1 # 비교 횟수 증가 (while)

def mergesort(array, info):
    
    _mergesort(array, 0, len(array)-1, info)

    return (info[0], info[1])

if __name__ == "__main__":
    array = [17, 9, 13, 8, 7, 7, -5, 6, 11, 3, 4, 1, 2]
    mergesort(array, [0, 0]);
    print(array)

