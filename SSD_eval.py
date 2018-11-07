#-*-coding:utf-8-*-
'''
Created on Nov 7,2018

@author: pengzhiliang
'''
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from model import SSD300
from dataset import ListDataset
# for decode
from encoder import DataEncoder

from PIL import Image

from display import dis_gt


resume_path = './checkpoint/ssd300_ckpt.pth'
print('Loading model...')
net = SSD300()
net.load_state_dict(torch.load(resume_path)['net'])
net.eval()

print('preparing image...')
img_path = './images/test_img.JPEG'
target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
bbox = np.array([11,37,90,105],dtype = np.float32)  
label = np.array([3],dtype = np.int32) 
# decode 
data_encoder = DataEncoder()

img = Image.open(img_path).convert('RGB')
images = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])(img.resize((300,300)))


with torch.no_grad():
    # (1,8732, 4) (1, 8732, 6) cpu
    loc_preds, conf_preds = net(images.unsqueeze(0))

    boxes, labels, scores = data_encoder.decode(loc_preds.cpu().data.squeeze(), 
                            F.softmax(conf_preds.cpu().squeeze(),dim=1).data,
                            score_thresh=0.01, nms_thresh=0.1)
    # 去掉背景
    # non_back = labels.nonzero().squeeze(0)
    # labels,boxes,scores = labels[non_back],boxes[non_back],scores[non_back]
    # 如果还有多个值，取得分最高的
    if len(labels)>1:
        max_score,index = scores.max(dim=0)
        labels = labels[index]
        boxes = boxes[index]
        scores = scores[index]
    print('finally :',boxes, labels, scores)
dis_gt(img,[target_classes[labels.item()]+':'+str(scores.item()),target_classes[label]],[boxes.numpy()*128,bbox])