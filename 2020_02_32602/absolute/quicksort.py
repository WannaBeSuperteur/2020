#!/usr/bin/python2
# source: https://github.com/ztgu/sorting_algorithms_py
# info: [비교 횟수, 이동 횟수]

def swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def partition(array, start, end, info):
    
    """ quicksort partitioning, using end """
    pivot = array[end]
    L = start
    R = end
    info[0] += 1 # 비교 횟수 증가 (while L<R)
    while L < R:
        
        info[0] += 1 # 비교 횟수 증가 (while array[L]<pivot)
        while array[L] < pivot:
            L += 1
            info[0] += 1 # 비교 횟수 증가 (while array[L]<pivot)

        info[0] += 1 # 비교 횟수 증가 (while array[R]>pivot)        
        while array[R] > pivot:
            R -= 1
            info[0] += 1 # 비교 횟수 증가 (while array[R]>pivot)
        
        info[1] += 1 # 이동 횟수 증가
        swap(array, L, R)
        # avoid hanging on the same numbers
        
        info[0] += 1 # 비교 횟수 증가 (if)
        if ( array[L] == array[R] ):
            L += 1

        info[0] += 1 # 비교 횟수 증가 (while L<R)
    return R

def _quicksort(array, start, end, info):

    """ Recursive quicksort function """
    info[0] += 1 # 비교 횟수 증가
    if start < end:
        split = partition(array, start, end, info)
        _quicksort(array, start, split-1, info)
        _quicksort(array, split+1, end, info)

def quicksort(array, info):
    
    _quicksort(array, 0, len(array)-1, info)

    return (info[0], info[1])

if __name__ == "__main__":
    array = [17, 9, 13, 8, 7, 7, -5, 6, 11, 3, 4, 1, 2]
    quicksort(array, [0, 0])
    print(array)

