import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np
import pandas as pd
import random
import re
import math
from os import listdir
from PIL import Image

# get array (training/test input) for images
def getArrForImages(rows, inputArr, inputDir, inputFiles):

    SIZE = 424
    CROP_SIZE = int(SIZE / 4)
    MARGIN = int(SIZE / 16)
    
    for i in range(rows):
        if i % 100 == 0: print(i)
        
        img = Image.open(inputDir + '/' + inputFiles[i])
        
        img = img.resize((CROP_SIZE, CROP_SIZE))
        img = img.crop((MARGIN, MARGIN, CROP_SIZE - MARGIN, CROP_SIZE - MARGIN))
        
        imgSeq = img.getdata()
        imgArr = np.array(imgSeq) # [[255, 0, 127], [255, 1, 126], ..., [7, 64, 255]]
        imgArr_ = []

        # convert into average of R, G and B
        for j in range(len(imgArr)):
            imgArr_.append(sum(imgArr[j]) / (3 * 255)) # 0 ~ 765 ->  0 ~ 1
            imgArr_[j] = 3 * imgArr_[j] - 1            # 0 ~ 1   -> -1 ~ 2
        
        inputArr.append(imgArr_)

if __name__ == '__main__':

    # SUBMISSION
    # prediction of 'survived' column

    # rows
    train_rows = 30000 # 61578
    test_rows = 30000 # 79975
    valid_rate = float(RD.loadArray('val_rate.txt')[0][0])

    # directories
    train_input_dir = 'images_training_rev1/images_training_rev1'
    train_output_file = 'training_solutions_rev1/training_solutions_rev1.csv'
    test_input_dir = 'images_test_rev1/images_test_rev1'

    # set data to validate (array 'isValid')
    isValid = []
    for i in range(train_rows):
        isValid.append(False)

    count = 0
    while count < train_rows * valid_rate:
        rand = random.randint(0, train_rows - 1)

        if isValid[rand] == False:
            count += 1
            isValid[rand] = True

    # training and test input
    train_input = []
    test_input = []

    # load training input images
    train_input_files = [f for f in listdir(train_input_dir)]
    getArrForImages(train_rows, train_input, train_input_dir, train_input_files)
    
    # load training output data
    train_output_data = np.array(RD.loadArray(train_output_file, ','))[1:, 1:]
    
    # load test input images
    test_input_files = [f for f in listdir(test_input_dir)]
    getArrForImages(test_rows, test_input, test_input_dir, test_input_files)
    
    # write training input
    RD.saveArray('train_input.txt', train_input, '\t', 1)

    # write training output
    RD.saveArray('train_output.txt', train_output_data, '\t', 1)

    # write test input
    RD.saveArray('test_input.txt', test_input, '\t', 1)
