# source: https://github.com/ztgu/sorting_algorithms_py

import bubblesort as BS
import heapsort as HS
import insertionsort as IS
import mergesort as MS
import quicksort as QS
import selectionsort as SS

# bubble sort
def bubble(array):
    result = BS.bubblesort(array, [0, 0])
    return result

# heap sort
def heap(array):
    result = HS.heap_sort(array, [0, 0])
    return result

# insertion sort
def insertion(array):
    result = IS.insertionsort(array, [0, 0])
    return result

# merge sort
def merge(array):
    result = MS.mergesort(array, [0, 0])
    return result

# quick sort
def quick(array):
    result = QS.quicksort(array, [0, 0])
    return result

# selection sort
def selection(array):
    result = SS.selectionsort(array, [0, 0])
    return result
