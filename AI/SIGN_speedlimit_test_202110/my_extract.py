from PIL import Image
import os
import numpy as np
import pandas as pd
import random

# extract images from 'signImage' directory
def extractImages():

    # try to create 'images' directory
    try:
        os.mkdir('images')
    except:
        pass

    train_count = 0
    test_count = 0
    
    # extract images
    for r, _, f in os.walk('signImage'):

        count = 0

        # for each file in /signImage
        for file in f:

            # set the label for each image
            if len(r.split('\\')) < 3: continue
            label0 = r.split('\\')[1] # 'Train' or 'Test'
            label1 = r.split('\\')[2] # 0, 1, ..., 42

            # speed limit sign: 0(20), 1(30), 2(50), 3(60), 4(70), 5(80), 7(100), 8(120)
            lab1 = int(label1)

            if not lab1 in [0, 1, 2, 3, 4, 5, 7, 8]:
                continue
            
            if lab1 == 0:
                new_class = 0
            elif lab1 == 1:
                new_class = 1
            elif lab1 == 2:
                new_class = 2
            elif lab1 == 3:
                new_class = 3
            elif lab1 == 4:
                new_class = 4
            elif lab1 == 5:
                new_class = 5
            elif lab1 == 7:
                new_class = 6
            elif lab1 == 8:
                new_class = 7

            label1 = str(new_class)

            # use only training image
            if label0 == 'Test': continue

            # do not use label0 because it is always 'Train'
            label = label1

            # print the process
            if count % 100 == 0: print(label, count)
            count += 1

            # mark as training or test data
            if file[len(file)-5] == '0' or file[len(file)-5] == '5':
                fileName = 'test_' + ('%05d' % test_count) + '_' + label + '.jpg'
                test_count += 1
            else:
                fileName = 'train_' + ('%05d' % train_count) + '_' + label + '.jpg'
                train_count += 1
            
            imgFile = os.path.join(r, file)

            # resize into 64*64 images
            # change image name: *.png to *.jpg
            # save image at 'images' folder
            pat = os.path.join(r, file)
            
            with Image.open(pat) as im:
                if im.size!=(64, 64):
                    im=im.resize((64, 64), Image.LANCZOS)

                im.save('images/' + fileName)
                
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
        if count % 70 == 0: print(str(count) + ' / ' + str(len(files)) + ', filename=' + str(file))
        count += 1

        # open each image file
        img = Image.open('images/' + file)
        pixel = img.load()

        # add label
        # 00_train, 01_train, ..., 42_train, 00_test, 01_test, ..., 42_test
        label_num = int(file.split('.')[0].split('_')[2])
        if 'test' in file: label_num += 100

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
