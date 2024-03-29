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
    for r, _, f in os.walk('myData'):

        count = 0

        # for each file in /myData
        for file in f:

            # label of image: 0, 1, ..., 42
            label = r.split('\\')[1]
            imgFile = os.path.join(r, file)

            # print the process
            if count % 100 == 0: print(label, count)
            count += 1

            # resize into 32*32 images
            # change image name: *.png to *.jpg
            # save image at 'images' folder
            pat = os.path.join(r, file)
            
            with Image.open(pat) as im:
                if im.size!=(32, 32):
                    im=im.resize((32, 32), Image.LANCZOS)
                im.save('images/' + label + ') ' + file.replace(".png",".jpg"))

# OFEC: output for each class
#       in the form of [class0output, class1output, ..., class42output]
def makeCsv(OFEC):

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
    
    imgSize = 32
    for i in range(imgSize):
        for j in range(imgSize):
            firstRow.append(str(i) + '_' + str(j))

    csvArray.append(firstRow)

    # convert each image into a row for the CSV file
    files = os.listdir('images')
    count = 0
    
    for file in files:

        # for count
        if count % 100 == 0: print(count, len(files))
        count += 1

        # open each image file
        img = Image.open('images/' + file)
        pixel = img.load()

        # add each pixel
        label = OFEC[int(file.split(')')[0])]
        thisRow = [label]
        
        for i in range(imgSize):
            for j in range(imgSize):
                thisRow.append(int(sum(pixel[i, j])/3))

        csvArray.append(thisRow)

    # save into file
    RD.saveArray('train_test.csv', csvArray, splitter=',', saveSize=1000)

# extract training data and test data
def extractTrainAndTest(testRate):

    # load the csv file including all the data (except for the first row)
    csvArray = RD.loadArray('train_test.csv', splitter=',')
    csvArray = csvArray[1:]
    numOfRows = len(csvArray)

    # write first row
    firstRow = ['label']
    
    imgSize = 32
    for i in range(imgSize):
        for j in range(imgSize):
            firstRow.append(str(i) + '_' + str(j))

    # designate training data and test data
    trainingData = [firstRow]
    testData = [firstRow]

    for i in range(numOfRows):
        rand = random.random()

        if rand >= testRate: trainingData.append(csvArray[i])
        else: testData.append(csvArray[i])

    # save file (trainingData and testData)
    RD.saveArray('train.csv', trainingData, splitter=',', saveSize=1000)
    RD.saveArray('test.csv', testData, splitter=',', saveSize=1000)

if __name__ == '__main__':
    
    # output for each class
    OFEC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 4, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0]

    # extract images from 'myData' directory
    extractImages()

    # make CSV file from the extracted images
    makeCsv(OFEC)

    # extract training data and test data
    extractTrainAndTest(0.15)
