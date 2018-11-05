#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
'''

#=================================================
#
#           Test tiny_vid_loader()
#
#=================================================
from __future__ import print_function
import torch,os,sys,random,cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from matplotlib import pyplot as plt
from encoder import DataEncoder
from torch.utils import data
from torchvision import transforms
from os.path import join as pjoin

import sys
sys.path.insert(0,'../dataset/')
sys.path.insert(0,'../')
from tiny_vid import *

if __name__ == '__main__':
    target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
    dst = tiny_vid_loader(transform = 'some augmentation')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, (img, gt_class,gt_bbox) in enumerate(trainloader):
        if i % 150 == 0:
            print(gt_class)
            print(gt_bbox)
            print(img)
            gt_class=gt_class.data.numpy()
            gt_bbox = gt_bbox.data.numpy()[0]
            inp = img.numpy()[0].transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            # np.clip(inp,0,1) inp中比0小的设为0，比1大设为1
            inp = np.clip(inp, 0, 1)*255
            inp = Image.fromarray(inp.astype('uint8')).convert('RGB')
            dis_gt(inp,target_classes[gt_class],gt_bbox)

            break