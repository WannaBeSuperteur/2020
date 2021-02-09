from PIL import Image
import os
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import random

# extract images from 'myData' directory
def extractImages():

    # try to create 'images' directory
    try:
        os.mkdir('images')
    except:
        pass
    
    # extract images
    for r, _, f in os.walk('VehicleImage'):

        count = 0

        # for each file in /VehicleImage
        for file in f:

            # set the label for each image
            label0 = r.split('\\')[1]
            label1 = r.split('\\')[2]

            if label0 == 'non-vehicles' and label1 == 'Far':
                label = 'NF'
            elif label0 == 'non-vehicles' and label1 == 'Left':
                label = 'NL'
            elif label0 == 'non-vehicles' and label1 == 'MiddleClose':
                label = 'NM'
            elif label0 == 'non-vehicles' and label1 == 'Right':
                label = 'NR'
            elif label0 == 'vehicles' and label1 == 'Far':
                label = 'VF'
            elif label0 == 'vehicles' and label1 == 'Left':
                label = 'VL'
            elif label0 == 'vehicles' and label1 == 'MiddleClose':
                label = 'VM'
            elif label0 == 'vehicles' and label1 == 'Right':
                label = 'VR'
            else:
                continue

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

    # write first row
    firstRow = ['label']
    
    imgWidth = 64
    imgHeight = 64
    for i in range(imgHeight):
        for j in range(imgWidth):
            firstRow.append(str(i) + '_' + str(j))

    csvArray.append(firstRow)

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
        OFEC_list = {'NF_train':100, 'NL_train':101, 'NM_train':102, 'NR_train':103,
                     'VF_train':110, 'VL_train':111, 'VM_train':112, 'VR_train':113,
                     'NF_test':120, 'NL_test':121, 'NM_test':122, 'NR_test':123,
                     'VF_test':130, 'VL_test':131, 'VM_test':132, 'VR_test':133}

        label = OFEC_list[file.split(')')[0]]
        if label % 20 >= 10: thisRow = [1] # a vehicle
        else: thisRow = [0] # not a vehicle
        labelList.append([label])

        # add each pixel
        for i in range(imgHeight):
            for j in range(imgWidth):
                thisRow.append(int(sum(pixel[j, i])/3))

        csvArray.append(thisRow)

    # save into file
    RD.saveArray('train_test.csv', csvArray, splitter=',', saveSize=50)
    RD.saveArray('label_list.csv', labelList)

# extract training data and test data
def extractTrainAndTest():

    # load the csv file including all the data (except for the first row)
    csvArray = RD.loadArray('train_test.csv', splitter=',')
    csvArray = csvArray[1:]
    numOfRows = len(csvArray)

    # load label list
    labelList = RD.loadArray('label_list.csv')

    # write first row
    firstRow = ['label']
    
    imgWidth = 64
    imgHeight = 64
    for i in range(imgHeight):
        for j in range(imgWidth):
            firstRow.append(str(i) + '_' + str(j))

    # designate training data and test data
    trainingData = [firstRow]
    testData = [firstRow]
    trainingLabel = []
    testLabel = []

    for i in range(numOfRows):

        # train or test
        if int(labelList[i][0]) >= 120: train = False
        else: train = True

        # car: 0, bus: 1
        csvArray[i][0] = str(int(csvArray[i][0]) % 2)

        # append to training/test data
        if train == True:
            trainingData.append(csvArray[i])
            trainingLabel.append(labelList[i])
        else:
            testData.append(csvArray[i])
            testLabel.append(labelList[i])

    # save file (trainingData and testData)
    RD.saveArray('train.csv', trainingData, splitter=',', saveSize=500)
    RD.saveArray('test.csv', testData, splitter=',', saveSize=500)
    RD.saveArray('trainLabels.csv', trainingLabel)
    RD.saveArray('testLabels.csv', testLabel)

if __name__ == '__main__':

    # extract images from 'myData' directory
    extractImages()

    # make CSV file (train_test.csv) from the extracted images
    makeCsv()

    # extract training data and test data (train.csv and test.csv)
    extractTrainAndTest()
