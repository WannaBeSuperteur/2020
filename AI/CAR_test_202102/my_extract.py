from PIL import Image
import os
import numpy as np
import pandas as pd
import random

# extract images from 'VehicleImage' directory
def extractImages():

    # try to create 'images' directory
    try:
        os.mkdir('images')
    except:
        pass

    train_count = 0
    test_count = 0
    
    # extract images
    for r, _, f in os.walk('VehicleImage'):

        count = 0

        # for each file in /VehicleImage
        for file in f:

            # set the label for each image
            label0 = r.split('\\')[1] # 'Train' or 'Test'
            label1 = r.split('\\')[2] # 0, 1, ..., 42

            if label0 == 'non-vehicles' and label1 == 'Far':
                label = '00'
            elif label0 == 'non-vehicles' and label1 == 'Left':
                label = '01'
            elif label0 == 'non-vehicles' and label1 == 'MiddleClose':
                label = '02'
            elif label0 == 'non-vehicles' and label1 == 'Right':
                label = '03'
            elif label0 == 'vehicles' and label1 == 'Far':
                label = '10'
            elif label0 == 'vehicles' and label1 == 'Left':
                label = '11'
            elif label0 == 'vehicles' and label1 == 'MiddleClose':
                label = '12'
            elif label0 == 'vehicles' and label1 == 'Right':
                label = '13'
            else:
                continue

            # print the process
            if count % 100 == 0: print(label, count)
            count += 1

            # mark as training or test data
            if file[len(file)-5] == '0' or file[len(file)-5] == '5':
                fileName = 'test_' + ('%05d' % test_count) + '_' + label[0] + '.jpg'
                test_count += 1
            else:
                fileName = 'train_' + ('%05d' % train_count) + '_' + label[0] + '.jpg'
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
        # 0_train, 0_test, 1_train, 1_test
        label_num = int(file.split('.')[0].split('_')[2])
        if 'test' in file: label_num += 100

        thisRow = [label_num % 100] # 0, 1
        labelList.append([label_num]) # 0 or 1 for train, 100 or 101 for test

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
