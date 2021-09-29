import numpy as np
import json
import tensorflow as tf
import tensorflow.keras.backend as K

from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import algo_readImgs as readImgs

# https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html
# https://github.com/marcotcr/lime/issues/453

def new_predict(img):

    imgSize = 64
    N = np.shape(img)[0]

    # convert to reshape
    new_img = np.reshape(img, (N*imgSize*imgSize, 3)) # (N*64*64, 3)
    new_img = list(new_img)

    #    [[5, 5, 5], [255, 255, 255], [3, 3, 3], ..., [17, 17, 17]] (N*64*64 elements)
    # -> [5, 255, 3, ..., 17]
    for pixel in range(N*imgSize*imgSize):
        new_img[pixel] = new_img[pixel][0]

    # reshape image into (N, 64*64)   
    new_img = np.reshape(new_img, (N, imgSize*imgSize))

    return model.predict(new_img)

def run_lime(start, end, model):

    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')

    # load images
    imgs = readImgs.readImgs(start, end)
    print('leng of imgs = ' + str(len(imgs)))
    imgSize = 64

    # reshape the image
    imgs = np.reshape(imgs, (end-start, imgSize*imgSize)).astype(float)

    print(np.shape(imgs))

    ###### LIME EXPLAINER ######
    # source: https://github.com/marcotcr/lime
    #         https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb
    for i in range(len(imgs)):

        print('test for image ' + str(start + i))
        
        img = imgs[i]

        # convert into list to reshape
        img = list(img)

        # reshape image into (64, 64, 3)
        for pixel in range(imgSize*imgSize):
            img[pixel] = [img[pixel], img[pixel], img[pixel]]

        img = np.array(img)
        img = np.reshape(img, (imgSize, imgSize, 3))
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img.astype('double'), new_predict, top_labels=1, hide_color=0, num_samples=1000)

        # explaination test
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.savefig('algo_1_lime_' + str(start + i) + '_0.png', bbox_inches='tight', pad_inches=0)

        # remove colorbar from image
        try:
            plt.colorbar().remove()
        except:
            pass
        
        #Select the same class explained on the figures above.
        ind = explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    
        plt.imshow(heatmap, cmap = 'RdBu', vmin = -heatmap.max(), vmax = heatmap.max())
        plt.savefig('algo_1_lime_' + str(start + i) + '_1.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    
    # load pre-trained model and choose two images to explain
    trainI = [[0]*4096]
    trainO = [[0]*2]

    model = tf.keras.models.load_model('carTest_model')

    run_lime(400, 405, model)
    run_lime(800, 805, model)
