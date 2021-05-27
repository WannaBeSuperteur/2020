import numpy as np
import json
import shap
import matplotlib.pyplot as plt
import visualize as v
import tensorflow as tf
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, '../../AI_BASE')
import deepLearning_GPU_helper as helper
import deepLearning_GPU as DGPU

# source: https://github.com/slundberg/shap

# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

def run_shap(start, end):

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')

    # load images
    interval = 5
    
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

    for i in range(int((end - start) / interval)):

        imgs_sub = imgs[i*interval : (i+1)*interval]
        
        e = shap.DeepExplainer(model, imgs_sub)
        shap_values = e.shap_values(imgs_sub)

        # reshape shap_values and imgs
        shap_values = np.reshape(shap_values, (2, interval, imgSize, imgSize))
        imgs_sub = np.reshape(imgs_sub, (interval, imgSize, imgSize, 1))

        # plot the feature attributions
        shap.image_plot(shap_values[0], -imgs_sub, show=False)
        plt.savefig('algo_0_shap_' + str(start + i * interval) + '_0.png')
        
        shap.image_plot(shap_values[1], -imgs_sub, show=False)
        plt.savefig('algo_0_shap_' + str(start + i * interval) + '_1.png')

if __name__ == '__main__':
    run_shap(350, 355)
    run_shap(850, 855)
