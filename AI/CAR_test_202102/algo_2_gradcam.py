# source: https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import visualize as v
import keras
import sys
import cv2

sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU_helper as helper
import deepLearning_GPU as DGPU

# https://github.com/jacobgil/keras-grad-cam/issues/17
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
    for var, grad in zip(var_list, grads)]

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    # saliency = K.gradients(K.sum(max_output), input_img)[0]
    saliency = _compute_gradients(K.sum(max_output), [input_img])[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    
    # g = tf.get_default_graph()
    g = tf.compat.v1.get_default_graph()
    
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x, multiple=0.1):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= multiple

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th': # https://github.com/keras-team/keras/issues/12649
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_no, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 2 # 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    
    conv_output = model.layers[0].layers[layer_no].output
    
    # grads = normalize(K.gradients(loss, conv_output)[0])
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (64, 64)) # (224, 224)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def gray_to_rgb(img):
    imgSize = 64
    newImg = np.zeros((64, 64, 3))
    
    for i in range(imgSize):
        for j in range(imgSize):
            for k in range(3):
                newImg[i][j][k] = img[0][i][j]

    return newImg

# main function
def run_gradcam(start, end):

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
    trainI = [[0]*4096]
    trainO = [[0]*2]

    f = open('car_model_config.txt', 'r')
    modelInfo = f.readlines()
    f.close()
    
    NN = helper.getNN(modelInfo, trainI, trainO) # Neural Network    
    op = helper.getOptimizer(modelInfo) # optimizer
    loss = helper.getLoss(modelInfo) # loss
    
    model = DGPU.deepLearningModel('model', op, loss, True)
    model.load_weights('model.h5')

    # predict using the model
    predictions = model.predict(imgs)

    print('shape of predictions = ' + str(np.shape(predictions)))

    for i in range(len(predictions)):
        print('image ' + str(start + i) + ' -> ' +
              str(round(helper.invSigmoid(predictions[i][0]) * 100, 4)) + '% / ' +
              str(round(helper.invSigmoid(predictions[i][1]) * 100, 4)) + '%')

    # explanation of image
    for i in range(len(predictions)):
        print('processing image ' + str(start + i))
        
        predicted_class = np.argmax(predictions[i])

        layerNos = [2, 4, 6, 8]
        layerNames = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]

        reshaped_img = imgs[i].reshape((1, imgSize, imgSize, 1))
        
        for j in range(4):
            cam, heatmap = grad_cam(model, reshaped_img, predicted_class, layerNos[j], layerNames[j])

            if gradcam_write == True:
                cv2.imwrite("gradcam_" + str(start + i) + "_" + layerNames[j] + ".jpg", cam)

        # convert (64, 64, 1) into (64, 64, 3)
        reshaped_img = reshaped_img.reshape((1, imgSize, imgSize))
        reshaped_img = gray_to_rgb(reshaped_img)
        reshaped_img = reshaped_img.reshape((1, imgSize, imgSize, 3))

        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(guided_model)
        saliency = saliency_fn([reshaped_img, 0])
        gradcam = saliency[0] * heatmap[..., np.newaxis]
        cv2.imwrite("guided_gradcam_" + str(start + i) + ".jpg", deprocess_image(gradcam, multiple))

if __name__ == '__main__':
    run_gradcam(350, 355)
    run_gradcam(850, 855)
