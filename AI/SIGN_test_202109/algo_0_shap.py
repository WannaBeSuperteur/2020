import numpy as np
import json
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import algo_readImgs as readImgs

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
    
    imgs = readImgs.readImgs(start, end)
    print('leng of imgs = ' + str(len(imgs)))
    imgSize = 64

    # reshape the image
    imgs = np.reshape(imgs, (end-start, imgSize*imgSize)).astype(float)

    print('shape of imgs = ' + str(np.shape(imgs)))

    # load pre-trained model
    trainI = [[0]*4096]
    trainO = [[0]*43]

    model = tf.keras.models.load_model('signTest_model')

    for i in range(int((end - start) / interval)):

        imgs_sub = imgs[i*interval : (i+1)*interval]

        print(np.array(imgs_sub))
        
        e = shap.DeepExplainer(model, imgs_sub)
        shap_values = e.shap_values(imgs_sub)

        # reshape shap_values and imgs
        shap_values = np.reshape(shap_values, (43, interval, imgSize, imgSize)) # 43 classes
        imgs_sub = np.reshape(imgs_sub, (interval, imgSize, imgSize, 1))

        # plot the feature attributions
        shap.image_plot(shap_values[0], -imgs_sub, show=False)
        plt.savefig('algo_0_shap_' + str(start + i * interval) + '_0.png')
        
        shap.image_plot(shap_values[1], -imgs_sub, show=False)
        plt.savefig('algo_0_shap_' + str(start + i * interval) + '_1.png')

if __name__ == '__main__':
    run_shap(200, 205)
    run_shap(500, 505)
    run_shap(1200, 1205)
