import main as m
import numpy as np
from PIL import Image

# read images
def readImages(train, start, end):

    # decide file name
    if train == True:
        fn = 'car_train_input.txt'
    else:
        fn = 'car_test_input.txt'

    # read image as numpy array
    return m.readCSV(fn, [0, 192*128], [start, end], '\t')

# visualize images from imgArray
# imgArray : contains image array with value 0.0 (black) ~ 1.0 (white)
# saveName : name of the image file
# show     : whether to show the image
def visualizeFromImgArray(imgArray, saveName, show):

    # for each image
    for i in range(len(imgArray)):
        if show == True: print('showing image ' + str(i) + '...')

        img = imgArray[i]

        # convert into [height][width][3] = [128][192][3] image
        converted = []
        for i in range(128):
            
            row = [] # each row of the image
            for j in range(192):

                pixel = [] # each pixel of the image
                for k in range(3): pixel.append(int(float(img[i * 192 + j]) * 255))
                row.append(pixel)

            converted.append(row)

        # visualize image
        convertedImg = Image.fromarray(np.array(converted).astype('uint8'), 'RGB')
        if show == True: convertedImg.show()

        # save image
        if saveName != None:
            convertedImg.save(saveName)

# visualize images
# need: car_train_input.txt / car_train_output.txt / car_test_input.txt / car_test_output.txt
# train    : from training data (True or False)
#            from test data when False
# start    : starting index
# end      : ending index
def visualizeImages(train, start, end):

    # read image as numpy array
    imgArray = readImages(train, start, end)

    # visualize images
    visualizeFromImgArray(imgArray, None, True)
    
if __name__ == '__main__':

    # visualize training image from index 234 to index 250
    visualizeImages(True, 234, 251)
