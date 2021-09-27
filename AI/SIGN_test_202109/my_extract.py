from PIL import Image
import os
import numpy as np
import pandas as pd
import random

# extract images from 'myData' directory
def extractImages():

    # try to create 'images' directory
    try:
        os.mkdir('images')
    except:
        pass
    
    # extract images
    for r, _, f in os.walk('signImage'):

        count = 0

        # for each file in /VehicleImage
        for file in f:

            # set the label for each image
            if len(r.split('\\')) < 3: continue
            label0 = r.split('\\')[1] # 'Train' or 'Test'
            label1 = r.split('\\')[2] # 0, 1, ..., 42

            # use only training image
            if label0 == 'Test': continue

            # 0, 1, ..., 42 -> 00, 01, ..., 42
            if len(label1) == 1: label1 = '0' + label1

            # do not use label0 because it is always 'Train'
            label = label1

            # print the process
            if count % 100 == 0: print(label, count)
            count += 1

            # mark as training or test data
            if file[len(file)-5] == '0' or file[len(file)-5] == '5':
                label += '_test'
            else:
                label += '_train'
            
            imgFile = os.path.join(r, file)

            # resize into 64*64 images
            # change image name: *.png to *.jpg
            # save image at 'images' folder
            pat = os.path.join(r, file)
            
            with Image.open(pat) as im:
                if im.size!=(64, 64):
                    im=im.resize((64, 64), Image.LANCZOS)
                im.save('images/' + label + ') ' + file.replace(".png",".jpg"))

# OFEC: output for each class
#       in the form of [class0output, class1output, ..., class42output]
def makeCsv():

    # write CSV file: for example, (10x10 images with label 0, 1, ..., or 7)
    
    # label 0,0 0,1 0,2 0,3 ... 0,9 1,0 ... 1,9 2,0 ... 9,9
    # 4     0   0   0   0   ... 0   0   ... 255 255 ... 0
    # 1     0   255 204 51  ... 0   51  ... 255 0   ... 0
    # 5     0   0   102 153 ... 0   0   ... 0   0   ... 135
    # 0     0   0   0   0   ... 65  173 ... 0   0   ... 0
    # 1     0   0   0   0   ... 102 200 ... 12  0   ... 0
    # ...   ... ... ... ... ... ... ... ... ... ... ... ...
    # 7     0   0   0   102 ... 102 225 ... 195 0   ... 0

    # initialize array
    csvArray = []
    imgWidth = 64
    imgHeight = 64

    # convert each image into a row for the CSV file
    files = os.listdir('images')
    count = 0
    labelList = []
    
    for file in files:

        # for count
        if count % 25 == 0: print(count, len(files))
        count += 1

        # open each image file
        img = Image.open('images/' + file)
        pixel = img.load()

        # add label
        # 00_train, 01_train, ..., 42_train, 00_test, 01_test, ..., 42_test
        file_label = file.split(')')[0]
        
        label_num = int(file_label[:2])
        if 'test' in file_label: label_num += 100

        thisRow = [label_num % 100] # 0, 1, ..., 42
        labelList.append([label_num]) # 0, 1, ..., 42 for train, 100, 101, ..., 142 for test

        # add each pixel
        for i in range(imgHeight):
            for j in range(imgWidth):
                thisRow.append(int(sum(pixel[j, i])/3))

        csvArray.append(thisRow)

    csvArray = np.array(csvArray)
    labelList = np.array(labelList)
    
    csvArray = pd.DataFrame(csvArray)
    labelList = pd.DataFrame(labelList)

    # save into file
    csvArray.to_csv('train_test.csv')
    labelList.to_csv('label_list.csv')

# extract training data and test data
def extractTrainAndTest():

    # load the csv file including all the data
    csvArray = pd.read_csv('train_test.csv', index_col=0)
    csvArray = np.array(csvArray)
    numOfRows = len(csvArray)

    # load label list
    labelList = pd.read_csv('label_list.csv', index_col=0)
    labelList = np.array(labelList)

    imgWidth = 64
    imgHeight = 64

    # designate training data and test data
    trainingData = []
    testData = []
    trainingLabel = []
    testLabel = []

    for i in range(numOfRows):

        # train or test
        if int(labelList[i][0]) >= 100: train = False
        else: train = True

        # append to training/test data
        if train == True:
            trainingData.append(csvArray[i])
            trainingLabel.append(labelList[i])
        else:
            testData.append(csvArray[i])
            testLabel.append(labelList[i])

    # save file (trainingData and testData)
    trainingData = pd.DataFrame(np.array(trainingData))
    testData = pd.DataFrame(np.array(testData))
    trainingLabel = pd.DataFrame(np.array(trainingLabel))
    testLabel = pd.DataFrame(np.array(testLabel))

    trainingData.to_csv('train.csv')
    testData.to_csv('test.csv')
    trainingLabel.to_csv('trainLabels.csv')
    testLabel.to_csv('testLabels.csv')

if __name__ == '__main__':

    # extract images from 'myData' directory
    extractImages()

    # make CSV file (train_test.csv) from the extracted images
    makeCsv()

    # extract training data and test data (train.csv and test.csv)
    extractTrainAndTest()
