import helper as h
import main as m
import visualize as v
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import math
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

import os
import gc

# crop a rectangular image (object) into m * n rectangular sub-images with same size
# and then fill (m_, n_)-th subimage with colors, and return the final image
# assumption: np.shape(imgArray) == (height, width, 3)
# pattern   : pattern to fill     (0   = BLACK       ,  1   = CHECK PATTERN)
# opacity   : opacity for filling (0.0 = transparent to 1.0                )
# genExplan : True for generating final explanation, False for just filling black
def cropAndFill(imgArray, m, n, m_, n_, pattern, opacity, genExplan):
    
    # get width and height of image
    imgWidth = len(imgArray[0])
    imgHeight = len(imgArray)

    if genExplan == True:
        imgArray_ = np.zeros((imgHeight, imgWidth, 3))
    else:
        imgArray_ = np.zeros((imgHeight, imgWidth))

    # fill as original
    for i in range(imgHeight):
        for j in range(imgWidth):
            if genExplan == True:
                for k in range(3):
                    try: # shape of imgArray is (height, width)
                        imgArray_[i][j][k] = imgArray[i][j]
                    except: # shape of imgArray is (height, width, 3)
                        imgArray_[i][j][k] = imgArray[i][j][k]
            else:
                imgArray_[i][j] = imgArray[i][j]

    # fill with black
    for i in range(int(m_ * imgHeight / m), int((m_+1) * imgHeight / m)):
        for j in range(int(n_ * imgWidth / n), int((n_+1) * imgWidth / n)):
            
            if pattern == 0 or (i + j) % 4 <= 1: # pattern == 0 (BLACK) or (i + j) % 2 == 0 (LINE PATTERN)
                if genExplan == True:
                    try:
                        imgArray_[i][j][0] = 0.6 * opacity + float(imgArray[i][j]) * (1.0 - opacity)
                        imgArray_[i][j][1] =                 float(imgArray[i][j]) * (1.0 - opacity)
                        imgArray_[i][j][2] = 1.0 * opacity + float(imgArray[i][j]) * (1.0 - opacity)
                    except:
                        imgArray_[i][j][0] = 0.6 * opacity + float(imgArray[i][j][0]) * (1.0 - opacity)
                        imgArray_[i][j][1] =                 float(imgArray[i][j][1]) * (1.0 - opacity)
                        imgArray_[i][j][2] = 1.0 * opacity + float(imgArray[i][j][2]) * (1.0 - opacity)
                else:
                    imgArray_[i][j] = float(imgArray[i][j]) * (1.0 - opacity)

    # return result
    return imgArray_

# distance between two vectors
def dist(vec0, vec1):
    assert(len(vec0) == len(vec1))
    
    result = 0
    for i in range(len(vec0)): result += pow(vec0[i] - vec1[i], 2)
    return math.sqrt(result)

# input image to the model and get values for each layer
# img     : image, array with value 0.0 (black) ~ 1.0 (white)
# option  : 'LHL' for last hidden layer and 'OUT' for output layer

# ref : https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
def layers(img, option, model):
    imgSize = 64

    # reshape image
    img = img.reshape((1, imgSize*imgSize))

    # input the image into the model, and get the last hidden layer
    if option == 'LHL':
        XAI_function_LHL = K.function([model.layers[0].input], [model.layers[14].output])
        layer_values_LHL = XAI_function_LHL(img) # shape: (1, 1, 40)
        return layer_values_LHL[0][0]

    # input the image into the model, get the output
    elif option == 'OUT':
        XAI_function_OUT = K.function([model.layers[0].input], [model.layers[15].output])
        layer_values_OUT = XAI_function_OUT(img) # shape: (1, 1, N) where N=43
        return layer_values_OUT[0][0]

# print and save heatmap
# index         : index of image
# layerLabel    : 'last hidden' or 'output'
# array         : array to print
# verboseOutput : show the image?
# subImgCountX  : sub-image count with X axis
def printAndSaveHeatmap(index, layerLabel, array, subImgCountX, verboseOutput):

    subimgs = len(array)
    neurons = len(array[0])

    # to display the value for each row
    # https://matplotlib.org/3.3.3/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(neurons))
    ax.set_yticks(np.arange(subimgs))

    # create x tick labels
    xLabels = []
    for i in range(neurons):
        if i % 5 == 0 or i == neurons-1 or neurons <= 10:
            xLabels.append(str(i))
        else:
            xLabels.append('')

    # create y tick labels
    yLabels = []
    for i in range(subimgs):
        mod = i % subImgCountX
            
        if mod == 0 or i == subimgs-1:
            yLabels.append(str(int(i / subImgCountX)) + ',' + str(int(mod)))
        else:
            yLabels.append('')

    # set x and y tick labels
    ax.set_xticklabels(xLabels)
    ax.set_yticklabels(yLabels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # show image
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(str(index).zfill(6) + '_' + layerLabel)
    if verboseOutput == True: plt.show()

# merge two images
# https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
# https://stackoverflow.com/questions/20361444/cropping-an-image-with-python-pillow
def mergeImgs(index, name0, name1, newName, marginLeft):

    image0 = Image.open(str(index).zfill(6) + '_' + name0 + '.png')
    image1 = Image.open(str(index).zfill(6) + '_' + name1 + '.png')

    # remove left white margin for each image
    # shape of image: (480, 640, 4)
    image0 = image0.crop((marginLeft[0], 0, 620, 480))
    image1 = image1.crop((marginLeft[1], 0, 620, 480))

    # save merged image
    newImage = Image.new('RGB', (image0.size[0] + image1.size[0], image0.size[1]))
    newImage.paste(image0, (0, 0))
    newImage.paste(image1, (image0.size[0], 0))
    newImage.save(str(index).zfill(6) + '_' + newName + '.png')

# XAI
# need: car_train_input.txt / car_train_output.txt / car_test_input.txt / car_test_output.txt
# train          : from training data (True or False)
#                  from test data when False
# imgNo          : image index
# imgArray       : test input image array
# model          : XAI model
# verboseOutput  : print text and image (about model output and last hidden layer)
# verboseXAI     : print XAI image
def XAI(train, imgNo, imgArray, model, verboseOutput, verboseXAI):

    # XAI with sub-images
    subimgWidth = 8
    subimgHeight = 8
    imgWidth = 64
    imgHeight = 64

    # imgWidth must be multiple of subimgWidth
    # imgHeight must be multiple of subimgHeight
    assert(imgWidth % subimgWidth == 0)
    assert(imgHeight % subimgHeight == 0)

    subImgCountX = int(imgWidth / subimgWidth)
    subImgCountY = int(imgHeight / subimgHeight)

    # read the list of test labels
    testLabels = np.array(pd.read_csv('testLabels.csv', index_col=0))
    
    # for each image
    for i in range(len(imgArray)):
        if i != imgNo: continue

        # close all images
        plt.close('all')
        
        print('running XAI with image with index ' + str(imgNo) + '...')

        # reshape the image
        img = imgArray[i]
        img = np.reshape(img, (imgHeight, imgWidth))

        # last hidden layer values for input of original image
        lastHidden_Original = layers(img, 'LHL', model)
        output_Original = layers(img, 'OUT', model)

        # number of neurons in the last hidden layer and output layer
        lastHidden_args = len(lastHidden_Original)
        output_args = len(output_Original)

        # fill-blacked images
        fillBlackedImages = []

        # last hidden and output layer of fill-blacked images
        lastHidden_FillBlacked = []
        output_FillBlacked = []

        # crop and fill black
        for j in range(subImgCountY):
            for k in range(subImgCountX):
                
                # get fill-blacked image
                filledImg = cropAndFill(img, subImgCountY, subImgCountX, j, k, pattern=0, opacity=1.0, genExplan=False)
                fillBlackedImages.append(filledImg)

                # last hidden layer values for input of fill-blacked image
                lastHidden_FillBlacked.append(layers(filledImg, 'LHL', model))
                output_FillBlacked.append(layers(filledImg, 'OUT', model))

        # print and show original image
        if verboseOutput == True:
            print('\nlast hidden layer : when input is original image:')
            print(lastHidden_Original)
            
        # print and show each fill-blacked image
        for j in range(subImgCountY):
            for k in range(subImgCountX):
               
                if verboseOutput == True:
                    print('\nlast hidden layer : when input is fill-blacked image (' + str(j) + ', ' + str(k) + '):')
                    print(lastHidden_FillBlacked[j * subImgCountX + k])
                
        # get last hidden layer and output difference between original image and fill-blacked images
        difs = []
        outputDifs = []
        if verboseOutput == True: print('\n <<< difs >>>')
        
        for j in range(subImgCountY):
            for k in range(subImgCountX):
                
                # get difference of last hidden layer
                dif = dist(lastHidden_Original, lastHidden_FillBlacked[j * subImgCountX + k])
                difs.append(dif)

                # get difference of output layer
                outputDif = dist(output_Original, output_FillBlacked[j * subImgCountX + k])
                outputDifs.append(outputDif)

                # print
                if verboseOutput == True:
                    print('\ndif (j=' + str(j) + ' k=' + str(k) + ') = ' + str(dif))
                    print('output: ' + str(output_FillBlacked[j * subImgCountX + k]))

        # transpose of difs
        difs_ = []
        outputDifs_ = []
        for j in range(subImgCountX * subImgCountY):
            difs_.append([difs[j]])
            outputDifs_.append([outputDifs[j]])

        # argmax of output
        output_FB_argmax = [[0]*output_args for _ in range(subImgCountX * subImgCountY)]
        
        for j in range(subImgCountX * subImgCountY):
            output_FB_argmax[j][np.argmax(output_FillBlacked[j])] = 1

        # print heatmap for last hidden layer (and difs)
        printAndSaveHeatmap(i, 'last_hidden', lastHidden_FillBlacked, subImgCountX, verboseOutput)
        printAndSaveHeatmap(i, 'difs_last_hidden', difs_, subImgCountX, verboseOutput)
        printAndSaveHeatmap(i, 'output', output_FillBlacked, subImgCountX, verboseOutput)
        printAndSaveHeatmap(i, 'argmax_output', output_FB_argmax, subImgCountX, verboseOutput)

        # merge images
        mergeImgs(i, 'last_hidden', 'difs_last_hidden', 'last_hidden', [140, 390])
        mergeImgs(i, 'output', 'argmax_output', 'output', [160, 160])

        # remove 'difs_last_hidden' and 'argmax_output' file
        os.remove(str(i).zfill(6) + '_difs_last_hidden.png')
        os.remove(str(i).zfill(6) + '_argmax_output.png')

        # save as file
        fileArray = []

        # columns for fileArray
        columns = []

        columns.append('subimg')
        for j in range(lastHidden_args):
            columns.append('LH_' + str(j))
        columns.append('')
        columns.append('dif')
        columns.append('')
        for j in range(output_args):
            columns.append('O_' + str(j))
        columns.append('')
        columns.append('outdif')
        columns.append('')
        for j in range(output_args):
            columns.append('maxO_' + str(j))
        columns.append('')
        columns.append('original argmax')
        
        if i == 0: columns_extended = ['imgNo', 'label'] + columns
        
        # info for each sub-image
        for j in range(subImgCountY * subImgCountX):

            Y = int(j / subImgCountX)
            X = j % subImgCountX
            
            resultOfThisSubImg = []

            resultOfThisSubImg.append(str(Y) + ',' + str(X))
            for k in range(lastHidden_args): resultOfThisSubImg.append(lastHidden_FillBlacked[j][k])
            resultOfThisSubImg.append('')
            resultOfThisSubImg.append(difs_[j][0])
            resultOfThisSubImg.append('')
            for k in range(output_args): resultOfThisSubImg.append(output_FillBlacked[j][k])
            resultOfThisSubImg.append('')
            resultOfThisSubImg.append(outputDifs_[j][0])
            resultOfThisSubImg.append('')
            for k in range(output_args): resultOfThisSubImg.append(output_FB_argmax[j][k])
            resultOfThisSubImg.append('')
            resultOfThisSubImg.append(np.argmax(output_Original))

            fileArray.append(resultOfThisSubImg)

        # initialize whole file array
        fileArray = pd.DataFrame(np.array(fileArray))
        fileArray.columns = columns
        fileArray.to_csv(str(i).zfill(6) + '_info.txt')
        
        # visualize the difference
        img_dif = copy.deepcopy(img)
        dif_max = max(difs)

        # fill with black with opacity (dif / dif_max)
        for j in range(subImgCountY):
            for k in range(subImgCountX):
                
                output_og = output_Original # output prediction for original image
                output_fb = output_FillBlacked[j * subImgCountX + k] # output prediction for fill-blacked image

                # opacity to fill each sub-image (prevent divide-by-zero error)
                if dif_max > 0:
                    opacity = 0.75 * difs[j * subImgCountX + k] / dif_max
                
                if dif_max == 0: # no difference
                    img_dif = cropAndFill(img_dif, subImgCountY, subImgCountX, j, k, pattern=0, opacity=0.0, genExplan=True)
                elif np.argmax(output_og) == np.argmax(output_fb): # max of output not changed
                    img_dif = cropAndFill(img_dif, subImgCountY, subImgCountX, j, k, pattern=0, opacity=opacity, genExplan=True)
                else: # max of output changed
                    img_dif = cropAndFill(img_dif, subImgCountY, subImgCountX, j, k, pattern=1, opacity=opacity, genExplan=True)

        # show image
        convertedImg = Image.fromarray(np.array(img_dif * 255.0).astype('uint8'), 'RGB')
        convertedImg.save('XAI_' + ('%05d' % i) + '.png')
                
# main
# visualizeImage : visualize images (including original / fill-blacked images)
# verboseOutput  : print text and image (model output and last hidden layer)
# verboseXAI     : print XAI image
if __name__ == '__main__':

    testRows = 7841
    matplotlib.use('Agg')

    # numpy print setting
    np.set_printoptions(edgeitems=30, linewidth=120)

    # test image array
    test_file = 'test.csv'
    test_data = np.array(pd.read_csv(test_file, index_col=0))
    test_input = np.round_(test_data[:, 1:] / 255.0, 3)

    # load model
    model = tf.keras.models.load_model('signTest_model')

    # run XAI with test images from index (start_index) to index (end_index)  
    for i in range(testRows // 50):
    
        # with 100% probability
        if random.random() < 1.0:
            XAI(False, i*50, test_input, model, verboseOutput=False, verboseXAI=False)
