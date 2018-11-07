#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
'''
from __future__ import print_function
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys
sys.path.insert(0,'../')

from encoder import DataEncoder
from PIL import Image, ImageDraw
from dataset.tiny_vid import *
from model.ssd import SSD300

if __name__ == '__main__':
    net = SSD300()
    # net.load_state_dict(torch.load('../checkpoint/ssd300_ckpt.pth')['net'])
    net.load_state_dict(torch.load('../pretrained/ssd.pth'))
    net.eval()

    data_encoder = DataEncoder()

    target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    trainset = ListDataset(root='/home/pzl/Data/tiny_vid', 
        list_file="/home/pzl/Data/tiny_vid/train_images.txt", train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
    for i, (images, loc_targets, conf_targets,_,__) in enumerate(trainloader):
        if i == 1 :
            # print(images,images.size()) # (1,3,300,300)
            # print(loc_targets,loc_targets.size()) # (1, 8732, 4)
            # print(conf_targets,conf_targets.size()) # (1, 8732)
            with torch.no_grad():
                loc, conf = net(images) 
                # print(loc.size(),conf.size())# (1,8732, 4) (1, 8732, 6)

            boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(dim=0),dim=1).data)
            print('box labels scores decode size:',boxes.size(), labels.size(), scores.size()) #(4,) () ()
            print(boxes, labels, scores)
            # draw = ImageDraw.Draw(img)
            # for box in boxes:
            #     box[::2] *= img.width
            #     box[1::2] *= img.height
            #     draw.rectangle(list(box), outline='red')
            # img.show()      


            break

