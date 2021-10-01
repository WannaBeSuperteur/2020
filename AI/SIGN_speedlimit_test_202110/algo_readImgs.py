import numpy as np
import pandas as pd

def readImgs(start, end):
    imgs = pd.read_csv('test.csv', index_col=0)
    imgs = np.array(imgs)[:, 1:] / 255.0
    return imgs[start:end]
