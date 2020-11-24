#!/usr/bin/python2
# source: https://github.com/ztgu/sorting_algorithms_py
# info: [비교 횟수, 이동 횟수]

def swap(array, i, j):

    i = int(i) # added
    j = int(j) # added
    
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def heapify(array, info):
    
    """ Build heap """
    # Middle in array
    start = (len(array) - 2) / 2
    info[0] += 1 # 비교 횟수 증가 (while)
    while start >= 0:
        perc_down(array, start, len(array) - 1, info)
        start -= 1
        info[0] += 1 # 비교 횟수 증가 (while)

def perc_down(array, start, end, info):
    
    """ Check/modify heap structure """
    largest = 2 * start + 1
    info[0] += 1 # 비교 횟수 증가 (while)
    while largest <= end:
        # left child < right child
        info[0] += 1 # 비교 횟수 증가 (if)
        # originally if (largest < end) and (array[largest] < array[largest + 1]):
        if (largest < end) and (array[int(largest)] < array[int(largest) + 1]):
            largest += 1
        # biggest child > parent
        info[0] += 1 # 비교 횟수 증가 (아래의 if)
        # originally if (array[largest] > array[start]):
        if (array[int(largest)] > array[int(start)]):
            info[1] += 1 # 이동 횟수 증가
            swap(array, largest, start)
            start = largest
            largest = 2 * start + 1
        else: 
            return

        info[0] += 1 # 비교 횟수 증가 (while)

def heap_sort(array, info):
    
    """ Sorting function """
    # biggest to smallest
    heapify(array, info)
    end = len(array) - 1

    info[0] += 1 # 비교 횟수 증가
    while end > 0:
        info[1] += 1 # 이동 횟수 증가
        # swap biggest node with end node
        swap(array, end, 0)
        # make sure first node is biggest
        perc_down(array, 0, end - 1, info)
        end -= 1

        info[0] += 1 # 비교 횟수 증가

    return (info[0], info[1])

if __name__ == "__main__":
    array = [17, 9, 13, 8, 7, -5, 6, 11, 3, 4, 1, 2]
    heap_sort(array, [0, 0])
    print(array)

