import numpy as np
import pandas as pd
from PIL import Image

# visualize images from imgArray
# imgArray : contains image array with value 0.0 (black) ~ 1.0 (white)
# saveName : name of the image file
# show     : whether to show the image
def visualizeFromImgArray(imgArray, imgLabels, imgIndex, show, optionalSaveName):

    width = 64
    height = 64

    # for each image
    if show == True:
        print('showing image ' + str(imgIndex) + '...')
    else:
        print('image ' + str(imgIndex) + '...')

    img = imgArray[imgIndex]

    # convert into [height][width][3] = [64][64][3] image
    converted = []
    for i in range(height):
            
        row = [] # each row of the image
        for j in range(width):

            pixel = [] # each pixel of the image
            for k in range(3): pixel.append(int(float(img[i * 64 + j]) * 255))
            row.append(pixel)

        converted.append(row)

    # visualize image
    convertedImg = Image.fromarray(np.array(converted).astype('uint8'), 'RGB')
    if show == True: convertedImg.show()

    # save image
    if optionalSaveName == None:
        saveName = 'signFewClassesTest_' + str(imgIndex) + '_' + str(imgLabels[imgIndex][0]) + '.png'
    else:
        saveName = optionalSaveName + '.png'
        
    convertedImg.save(saveName)
    
if __name__ == '__main__':

    # save grayscaled every 50th test images
    imgArray = np.round_(np.array(pd.read_csv('test.csv', index_col=0)) / 255.0, 3)
    imgLabels = np.array(pd.read_csv('testLabels.csv', index_col=0))
    imgs = len(imgArray)

    for i in range(imgs // 50):
        visualizeFromImgArray(imgArray, imgLabels, i*50, False, None)
