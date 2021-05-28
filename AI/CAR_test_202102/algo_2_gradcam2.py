# https://www.machinecurve.com/index.php/2019/11/28/visualizing-keras-cnn-attention-grad-cam-class-activation-maps/

# =============================================
# Grad-CAM code
# =============================================
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import visualize as v

import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU_helper as helper
import deepLearning_GPU as DGPU

def gray_to_rgb(img):
    imgSize = 64
    newImg = np.zeros((64, 64, 3))
    
    for i in range(imgSize):
        for j in range(imgSize):
            for k in range(3):
                newImg[i][j][k] = img[0][i][j]

    return newImg

# modified model with Input Shape = (None, 64, 64, 1)
#Layer (type)                 Output Shape              Param #
#=================================================================
#conv2d (Conv2D)              (None, 62, 62, 64)        640
#_________________________________________________________________
#max_pooling2d (MaxPooling2D) (None, 31, 31, 64)        0
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, 29, 29, 64)        36928
#_________________________________________________________________
#max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 12, 12, 64)        36928
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 4, 4, 32)          18464
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 512)               0
#_________________________________________________________________
#dense (Dense)                (None, 40)                20520
#_________________________________________________________________
#dropout (Dropout)            (None, 40)                0
#_________________________________________________________________
#dense_1 (Dense)              (None, 40)                1640
#_________________________________________________________________
#dropout_1 (Dropout)          (None, 40)                0
#_________________________________________________________________
#dense_2 (Dense)              (None, 40)                1640
#_________________________________________________________________
#dense_3 (Dense)              (None, 2)                 82
#=================================================================
def modified_model():
    inputs = tf.keras.Input(shape=(64, 64, 1))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(40, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(40, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(40, activation='relu', name='last_hidden_layer')(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid', name='output_layer')(x)

    return tf.keras.Model(inputs, outputs)

# main function
def run_gradcam(start, end, input_class):

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')

    # config
    gradcam_write = False
    multiple = 0.15

    # https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta/66222183
    tf.compat.v1.disable_eager_execution()

    # load images
    imgs = v.readImages(False, start, end)
    print('leng of imgs = ' + str(len(imgs)))
    imgSize = 64

    # reshape the image
    imgs = np.reshape(imgs, (end-start, imgSize*imgSize)).astype(float)

    print('shape of imgs = ' + str(np.shape(imgs)))

    # load pre-trained model
    trainI = [[0]*imgSize*imgSize]
    trainO = [[0]*2]

    f = open('car_model_config.txt', 'r')
    modelInfo = f.readlines()
    f.close()
    
    NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
    op = helper.getOptimizer(modelInfo) # optimizer
    loss = helper.getLoss(modelInfo) # loss
    
    model = DGPU.deepLearningModel('model', op, loss, True)
    model.load_weights('model.h5')
    model.summary()

    # modify model
    newModel = modified_model()
    newModel.summary()

    # load weights
    temp_weights = [layer.get_weights() for layer in model.layers]
    
    for i in range(len(newModel.layers)-1):
        newModel.layers[i+1].set_weights(temp_weights[i+2])

    newModel.build((None, imgSize, imgSize, 1))
    newModel.summary()

    # Find the index of the to be visualized layer above
    layer_index = utils.find_layer_idx(newModel, 'output_layer')
    newModel.layers[layer_index].activation = tf.keras.activations.linear
    newModel = utils.apply_modifications(newModel)
    newModel.summary()

    # print predictions
    print(' < predictions for each image >')

    N = end - start
    imgs_reshaped = imgs.reshape(N, imgSize, imgSize, 1)
    predictions = newModel.predict(imgs_reshaped)

    print('shape of predictions = ' + str(np.shape(predictions)))

    # DO NOT apply invSigmoid for the predictions of newModel
    for i in range(len(predictions)):
        print('image ' + str(start + i) + ' -> ' +
              str(round(predictions[i][0] * 100, 4)) + '% / ' +
              str(round(predictions[i][1] * 100, 4)) + '%')

    # explanation of image
    print(' < explanations for each image >')
    
    for i in range(len(imgs)):

        # reshape input image
        img = imgs[i].reshape(imgSize, imgSize, 1)

        # Matplotlib preparations
        fig, axes = plt.subplots(1, 3)

        # Generate visualization
        visualization = visualize_cam(newModel, layer_index, filter_indices=input_class[i], seed_input=img)

        axes[0].imshow(img[..., 0], cmap='gray') 
        axes[0].set_title('Input')
        axes[1].imshow(visualization)
        axes[1].set_title('Grad-CAM')

        heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
        original = np.uint8(cm.gray(img[..., 0])[..., :3] * 255)

        axes[2].imshow(overlay(heatmap, original))
        axes[2].set_title('Overlay')

        fig.suptitle(f'MNIST target = {input_class}')
        plt.savefig('algo_2_gradcam2_' + str(start + i) + '.png')

if __name__ == '__main__':
    run_gradcam(350, 355, [0, 0, 0, 0, 0])
    run_gradcam(850, 855, [1, 1, 1, 1, 1])
