import os
import random
import math
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json
from keras import backend as K
from PIL import Image

import matplotlib.pyplot as plt

## save cropped image with specified pixels for top, bottom, left and right
# location    : location where train and test images exist
# file_list   : list of image files to crop
# width       : width of each element in the array
# height      : height of each element in the array
# top         : crop rows at the top N pixels
# bottom      : crop rows at the bottom N pixels
# left        : crop columns at the left N pixels
# right       : crop columns at the right N pixels
# minWidth    : minimum width of cropped image
# minHeight   : minimum height of cropped image
# i           : changing value to prevent duplicated file name
def cropImgs(location, file_list, top, bottom, left, right, minWidth, minHeight, i):
    for file in range(len(file_list)): # for each image
        im = Image.open(location + file_list[file])
        width = im.size[0] # width of image
        height = im.size[1] # height of image

        # constraint of minWidth and minHeight
        if width-(left+right) < minWidth or height-(top+bottom) < minHeight: continue

        # crop
        im = im.crop((left, top, width-right, height-bottom))

        # save: XXX.yyy -> XXX(number).yyy (eg: 1_01.png -> 1_0149.png)
        im.save(location + file_list[file].split('.')[0] + str(file + i * len(file_list)) +
                '.' + file_list[file].split('.')[1])

## return shape-changed image array([height][width]) of original iamge array([height][width][3])
# imArray    : original image array
# discrete   : element of returning array contains integer between -4.0 ~ +5.0, otherwise sum of R, G and B
def imArrayToArray(imArray, discrete):

    # height and width
    height = len(imArray)
    width = len(imArray[0])

    # [height][width] array of this image
    array = [[0]*width for _ in range(height)]

    # shape change: [height][width][3] -> [height][width]
    # value (-4.0 ~ +5.0) based on brightness : sum of R, G and B values
    # (-4.0 : BLACK, 0.0 : GRAY, +5.0 : WHITE)
    # [Note: discrete input value is better for training than continuous input value]
    for i in range(len(array[0])):
        for j in range(len(array)):
            if discrete: array[j][i] = int(sum(imArray[j][i]) / 77) - 4
            else: array[j][i] = sum(imArray[j][i])

    return array

## return array that represents each image in location
## each element of array represent each image with width 'width' and height 'height'
# location   : location where train and test images exist
# width      : width of each element in the array
# height     : height of each element in the array
def loadImgs(location, width, height):
    file_list = os.listdir(location)

    imgArray = [] # array to return
    labels = [] # labels to return

    for file in range(len(file_list)):
        im = Image.open(location + file_list[file])
        im = im.resize((width, height))
        
        imArray = np.array(im) # [height][width][3] numpy array of this image
        array = imArrayToArray(imArray, True)

        # append this array to imgArray
        imgArray.append(array)
        labelArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # index of that number = 1, other = 0
        labelArray[int(file_list[file][0])] = 1
        labels.append(labelArray) # label is ALWAYS equal to first character of file name

    return [imgArray, labels, file_list]

## save array as image
# imArray     : shape is [height][width][3]
# fileName    : file name of image to save
# showImg     : show image?
def saveImg(imArray, fileName, showImg):
    img = Image.fromarray(imArray, 'RGB')
    img.save(fileName)
    if showImg == True: img.show()
