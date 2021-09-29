# from ExplainableCNN/Code/TrainCifar.py of ref
#      ExplainableCNN/Code/Test.py of ref

# ref: https://github.com/tavanaei/ExplainableCNN/tree/master/Code

import argparse
from algo_3_XCNN_XAI import *
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import algo_3_XCNN_Test as Test # added

import tensorflow as tf
import pandas as pd
import cv2

# https://stackoverflow.com/questions/67724544/runtimeerror-cuda-error-invalid-device-ordinal
# https://stackoverflow.com/questions/67730572/keyerror-cuda-visible-devices
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=5, help='training epoch') # default=300
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-wd', type=float, default=0.000001,help='weight decay')
parser.add_argument('-beta', type=float, default=0.9, help='momentum')
parser.add_argument('-cuda', action='store_true', help='Cuda operation')
parser.add_argument('-batch', type=int, default=128, help='batch_size')
parser.add_argument('-lr_decay', type=float, default=.99, help='lr decay')
parser.add_argument('-gpu', type=int, default=1, help='GPU device')

opt = parser.parse_args()
print(opt)

#transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])

#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])

#trainset = torchvision.datasets.CIFAR10(
#    root='cifar_data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=opt.batch, shuffle=True, num_workers=2)
# (to remove BrokenPipeError: [Errno 32] Broken pipe)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=opt.batch, shuffle=True, num_workers=0)

#testset = torchvision.datasets.CIFAR10(
#    root='cifar_data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(
#    testset, batch_size=opt.batch//2, shuffle=False, num_workers=2)
# (to remove BrokenPipeError: [Errno 32] Broken pipe)
#testloader = torch.utils.data.DataLoader(
#    testset, batch_size=opt.batch//2, shuffle=False, num_workers=0)

classes = ('class00', 'class01', 'class02', 'class03', 'class04',
           'class05', 'class06', 'class07', 'class08', 'class09',
           'class10', 'class11', 'class12', 'class13', 'class14',
           'class15', 'class16', 'class17', 'class18', 'class19',
           'class20', 'class21', 'class22', 'class23', 'class24',
           'class25', 'class26', 'class27', 'class28', 'class29',
           'class30', 'class31', 'class32', 'class33', 'class34',
           'class35', 'class36', 'class37', 'class38', 'class39',
           'class40', 'class41', 'class42')

# for XAI, from ExplainableCNN/Code/Test.py of ref
def visualize(ims, heatmaps, thrmaps, name):
    plt.figure(figsize=(24,6))
    n = len(heatmaps)
    
    for i in range(n):
        plt.subplot(3,n,i+1)
        plt.imshow(ims[i])
        plt.axis('off')

    for i in range(n):
        plt.subplot(3,n,n+i+1)
        plt.imshow(heatmaps[i],cmap='seismic')
        plt.axis('off')
            
    if len(thrmaps)>0:
        for i in range(n):
            plt.subplot(3,n,2*n+i+1)
            plt.imshow(thrmaps[i])
            plt.axis('off')
                
    #plt.show()
    plt.savefig(name + '.png')

# from ExplainableCNN/Code/Test.py of ref
def vis(maps, data, start, count):
    kernel=np.ones([3,3],dtype=np.uint8)
    heatmaps = []
    thrmaps = []
    for i in range(start, start + count): # min(len(maps),100)
        if len(maps[i].shape)>2:
            tmp = np.transpose(maps[i],(1,2,0)).squeeze()
        else:
            tmp = maps[i].squeeze()
        tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
        tmp = 1-(tmp)
        heatmaps.append(tmp.copy())

        print('i, maps[i]:')
        print(i)
        print(np.shape(maps[i]))
        print(maps[i])
        
        ## Thresholding for Localization (Future work)
        if maps[i].shape[0]>1:
            tmp2 = tmp.copy()
            tmp2[tmp2>0.5]=1
            tmp2[tmp2<=0.5]=0
            tmp2 = cv2.dilate(tmp2,kernel,iterations=1)
            tmp2 = cv2.erode(tmp2,kernel,iterations=1)
            tmp2 = tmp2*255
            gray = tmp2.astype('uint8')
            ret,thresh1 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
            contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            print('i, data[i]:')
            print(i)
            print(np.shape(data[i]))
            print(data[i])
            
            im = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
            box = None
            c = 0
            for ind, cont in enumerate(contours):
                if cont.shape[0]>c:
                    c = cont.shape[0]
                    box = cont.copy()

            elps = cv2.boundingRect(box)
            cv2.rectangle(im,elps,(0,0,0),1)
            thrmaps.append(im[:,:,::-1])
    return heatmaps,thrmaps

with tf.device('/gpu:0'):

    # model
    net = tf.keras.models.load_model('signTest_model')

    print('------ net info ------')
    print(net)
    print('----------------------')

    criterion = nn.CrossEntropyLoss()
    decay_step = opt.epoch//2

    # load data (images and labels)
    # from car_test_input.txt and car_test_output.txt

    # column 0 and column 1 of 'labels' and 'prediction' means
    # 'not car' and 'car', respectively
    inputs = (np.array(pd.read_csv('test.csv', index_col=0)).astype(float))[:, 1:] / 255.0
    labels = np.array(pd.read_csv('signTest_groundTruth.csv', index_col=0)).astype(float)
    predictions = np.array(pd.read_csv('signTest_prediction.csv', index_col=0)).astype(float)

    print(np.shape(inputs))
    print(inputs)
    print(np.shape(labels))
    print(labels)
    print(np.shape(predictions))
    print(predictions)

    # for XAI
    #maps = net.maps.cpu()
    maps = np.reshape(inputs, (-1, 1, 64, 64))
    name = 'XAI_Sign'

    for startAndCount in [[200, 5], [500, 5], [1200, 5]]:
        
        start = startAndCount[0]
        count = startAndCount[1]
        heatmaps, thrmaps = vis(maps, inputs, start, count)

        # input shape change
        # original trainCifar.py -> np.shape(ims) = (N, 32, 32, 3)
        inputs = np.reshape(inputs, (-1, 64, 64, 1))

        # XAI
        print('shape of input:')
        print(np.shape(inputs))
        visualize(inputs[start:start+count], heatmaps, thrmaps, name + '_' + str(start))
