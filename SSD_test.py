#-*-coding:utf-8-*-
'''
Created on Nov 6,2018

@author: pengzhiliang
'''
from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from model import SSD300
from dataset import ListDataset
from loss import SSDLoss
# for decode
from encoder import DataEncoder

from PIL import Image

def compute_iou_acc(in_class,gt_class,in_coor,gt_coor,theta=0.5):
    """
    设置阈值为0.5  <0.5不将其算入分类准确率
    参数：
        in_class: 预测的class分类,(n,) tensor
        in_coor : 预测坐标, (n,4) tensor
        gt: groundtruth (n,5) tensor
    返回：
        分类准确个数，IoU
    """
    # print('input metrics:',in_coor,gt_coor)
    in_coor[in_coor<0] = 0
    in_coor[in_coor>1] = 1


    xA = torch.max(in_coor[0], gt_coor[0]) #(N,)
    yA = torch.max(in_coor[1], gt_coor[1]) #(N,)
    xB = torch.min(in_coor[2], gt_coor[2]) #(N,)
    yB = torch.min(in_coor[3], gt_coor[3]) #(N,)

    if xA > xB: # 无交集
        interArea = 0
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (in_coor[2] - in_coor[0] + 1) * (in_coor[3] - in_coor[1] + 1)
    boxBArea = (gt_coor[2] - gt_coor[0] + 1) * (gt_coor[3] - gt_coor[1] + 1)
    IoU = interArea / (boxAArea + boxBArea - interArea)

    # print(IoU,'\n',in_class,'\n',gt_class,'\n',in_coor,'\n',gt_coor)
    # class_acc = 1 if gt_class.item() == in_class.item() else 0
    class_acc = 1 if (gt_class.item() == in_class.item()) and IoU >theta else 0
    # print(class_acc,mean_IoU)
    return class_acc,IoU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size =1
data_encoder = DataEncoder()

print('Loading model..')
net = SSD300()
net.load_state_dict(torch.load('./checkpoint/ssd300_ckpt.pth')['net'])
net.to(device)
net.eval()

print('Preparing dataset..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
testset = ListDataset(root='/home/pzl/Data/tiny_vid', list_file="/home/pzl/Data/tiny_vid/test_images.txt", train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)



acc = 0.
mean_iou = 0.
with torch.no_grad():
    # batch_size = 1
    for batch_idx, (images, loc_targets, conf_targets,real_boxes,real_class) in enumerate(testloader):
        # (1,3,300,300) cpu--> gpu
        images = images.to(device)
        #  (1, 8732, 4) (1, 8732) cpu
        loc_targets ,conf_targets= loc_targets ,conf_targets
        #  (1,4) (1,) cpu
        real_boxes ,real_class = real_boxes.squeeze() ,real_class.squeeze()
        # (1,8732, 4) (1, 8732, 6) gpu
        loc_preds, conf_preds = net(images)
        # decode([8732,4] , [8732,6]) gpu --> cpu
        boxes, labels, scores = data_encoder.decode(loc_preds.cpu().data.squeeze(), 
                                F.softmax(conf_preds.cpu().squeeze(),dim=1).data,
                                score_thresh=0.01, nms_thresh=0.1)
        # print('\n\n',batch_idx+1,' test iter:')
        # print('pred:',boxes, labels, scores)
        # print('ground truth:',real_boxes,real_class)
        # 去掉背景
        # non_back = labels.nonzero().squeeze(0)
        # labels,boxes,scores = labels[non_back],boxes[non_back],scores[non_back]
        # 如果还有多个值，取得分最高的
        if len(labels)>1:
            max_score,index = scores.max(dim=0)
            labels = labels[index]
            boxes = boxes[index]
        # print('finally :',boxes, labels, scores)
        class_acc,IoU = compute_iou_acc(labels,real_class,boxes.squeeze(),real_boxes)# background class = 0
        acc += class_acc
        mean_iou += IoU
    print('Mean IoU: %.3f  Classificatin Acc: %.3f (%d/%d)' 
                %(mean_iou/(batch_idx+1),1.*acc/(batch_idx+1),acc,batch_idx+1))

