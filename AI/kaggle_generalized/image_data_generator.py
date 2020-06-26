import os
import numpy as np
from PIL import Image

# return the image as a line written to the *.csv file (without '\n' at the end)
# with max value +1.0 and min value -1.0

# imageFileName: the name of image file (e.g. imageDir/filename.png)
#              #*..
# for example, ###* -> "(label),1,0,-1,-1,1,1,1,0,-1,0,1,1" [train] (label!=None)
#              .*##    "1,0,-1,-1,1,1,1,0,-1,0,1,1" [test] (label==None)
def makeImageAsLine(imageFileName, width, height, RW, GW, BW, label):

    im = Image.open(imageFileName)
    
    originalWidth = im.size[0] # width of image
    originalHeight = im.size[1] # height of image
    im = im.resize((width, height)) # resize image
    imArray = np.array(im) # [height][width][3] numpy array of this image

    # initialize result
    result = ''
    if label != None: result = label + ','

    # [height][width] array of this image
    array = [[0]*width for _ in range(height)]

    # shape change: [height][width][3] -> [height][width]
    for i in range(len(array[0])):
        for j in range(len(array)):
            array[j][i] = (imArray[j][i][0] * RW + imArray[j][i][1] * GW + imArray[j][i][2] * BW) / (RW + GW + BW) # 0.0 ~ 255.0
            array[j][i] /= 255.0 # 0.0 ~ 1.0
            array[j][i] = array[j][i] * 2.0 - 1.0 # -1.0 ~ 1.0

            result += str(round(array[j][i], 3)) + ','

    return result[:len(result)-1]
    
# create *.csv file using makeImageAsLine

# configFileName: configuration file name
# in the form of ...
#
# width height RW GW BW trainImagesFolder testImagesFolder
# keyword0 value0
# keyword1 value1
# ...
# keywordN valueN
def makeCSV(configFileName):

    f = open(configFileName, 'r')
    flines = f.readlines()
    f.close()

    # width height RW GW BW
    baseInfo = flines[0].split('\n')[0].split(' ')
    
    width = int(baseInfo[0]) # width of each image
    height = int(baseInfo[1]) # height of each image
    RW = float(baseInfo[2]) # weight for RED
    GW = float(baseInfo[3]) # weight for GREEN
    BW = float(baseInfo[4]) # weight for BLUE

    # list of training and test images
    trainDir = baseInfo[5] # train images folder
    testDir = baseInfo[6] # test images folder
    trainImgs = os.listdir(trainDir) # file names of train images folder
    testImgs = os.listdir(testDir) # file names of train images folder

    # content of CSV files
    trainCSVFile = 'label,' # content of train CSV file
    testCSVFile = '' # content of test CSV file

    for i in range(width*height):
        if i < width*height-1:
            trainCSVFile += 'col' + str(i) + ','
            testCSVFile += 'col' + str(i) + ','
        else:
            trainCSVFile += 'col' + str(i) + '\n'
            testCSVFile += 'col' + str(i) + '\n'

    # convert train images (training_image.csv)
    for i in range(len(trainImgs)):
        if i % 50 == 0: print('training image ' + str(i))
        
        for j in range(1, len(flines)):
            flinesSplit = flines[j].split('\n')[0].split(' ')
            
            if flinesSplit[0] in trainImgs[i]:
                label = flinesSplit[1]
                trainCSVFile += makeImageAsLine(trainDir+'/'+trainImgs[i], width, height, RW, GW, BW, label) + '\n'
                break

    f = open('training_image.csv', 'w')
    f.write(trainCSVFile)
    f.close()

    # convert test images (test_image.csv)
    for i in range(len(testImgs)):
        if i % 50 == 0: print('testing image ' + str(i))
        
        testCSVFile += makeImageAsLine(testDir+'/'+testImgs[i], width, height, RW, GW, BW, None) + '\n'

    f = open('test_image.csv', 'w')
    f.write(testCSVFile)
    f.close()

if __name__ == '__main__':
    makeCSV('input_output_image_info.txt')
