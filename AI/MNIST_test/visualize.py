import main as m
import numpy as np
from PIL import Image

# read images
def readImages(train, start, end):

    # decide file name
    if train == True:
        fn = 'mnist_train_input.txt'
    else:
        fn = 'mnist_test_input.txt'

    # read image as numpy array
    return m.readCSV(fn, [0, 28*28], [start, end], '\t')

# visualize images from imgArray
# imgArray : contains image array with value 0.0 (black) ~ 1.0 (white)
def visualizeFromImgArray(imgArray):

    # for each image
    for i in range(len(imgArray)):
        print('showing image ' + str(i) + '...')

        img = imgArray[i]

        # convert into [height][width][3] = [28][28][3] image
        converted = []
        for i in range(28):
            
            row = [] # each row of the image
            for j in range(28):

                pixel = [] # each pixel of the image
                for k in range(3): pixel.append(int(float(img[i * 28 + j]) * 255))
                row.append(pixel)

            converted.append(row)

        # visualize image
        convertedImg = Image.fromarray(np.array(converted).astype('uint8'), 'RGB')
        convertedImg.show()

# visualize images
# need: mnist_train_input.txt / mnist_train_output.txt / mnist_test_input.txt / mnist_test_output.txt
# train    : from training data (True or False)
#            from test data when False
# start    : starting index
# end      : ending index
def visualizeImages(train, start, end):

    # read image as numpy array
    imgArray = readImages(train, start, end)

    # visualize images
    visualizeFromImgArray(imgArray)
    
if __name__ == '__main__':

    # visualize training image from index 234 to index 250
    visualizeImages(True, 234, 251)
