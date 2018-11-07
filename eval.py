#-*-coding:utf-8-*-
'''
Created on Nov 7,2018

@author: pengzhiliang
'''
from __future__ import print_function
import numpy as np

import torch
from PIL import Image
from model import Net
from dataset import xywh_to_x1y1x2y2, x1y1x2y2_to_xywh
from torchvision import transforms
from display import dis_gt

resume_path = '/home/pzl/object-localization/checkpoint/best_model.pkl'
print('Loading model...')
net = Net('mobilenet',freeze_basenet = False)
net.load_state_dict(torch.load(resume_path)["model_state"])
net.eval()

print('load image...')
img_path = './images/test_img.JPEG'
target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
bbox = np.array([11,37,90,105],dtype = np.float32)  
label = np.array([3],dtype = np.int32) 

img = Image.open(img_path).convert('RGB')
IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])(img)
# gt_class,gt_bbox =torch.Tensor(label),torch.Tensor(bbox/128.)

with torch.no_grad():
    outputs_reg,outputs_class = net(IMG.unsqueeze(0))
    # print(outputs_reg,outputs_class)
    _,pred_label = outputs_class.squeeze(0).max(dim = 0)
    pred_bbox = xywh_to_x1y1x2y2(outputs_reg).squeeze(0)
    print(pred_label,pred_bbox)

dis_gt(img,[target_classes[int(pred_label.item())],target_classes[label]],[pred_bbox.numpy()*128,bbox])
