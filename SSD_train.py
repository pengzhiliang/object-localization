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
# from utils import progress_bar
from dataset import ListDataset
from loss import SSDLoss
from metrics import compute_iou_acc
# for decode
from encoder import DataEncoder


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
batch_size = 30
num_workers = 12
end_epoch = 200
# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='/home/pzl/Data/tiny_vid', list_file="/home/pzl/Data/tiny_vid/train_images.txt", train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = ListDataset(root='/home/pzl/Data/tiny_vid', list_file="/home/pzl/Data/tiny_vid/test_images.txt", train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Model
net = SSD300().to(device)
if args.resume:
    print('====> Resuming from checkpoint..\n')
    checkpoint = torch.load('./checkpoint/ssd300_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # Convert from pretrained VGG model.
    net.load_state_dict(torch.load("./pretrained/ssd.pth"))

criterion = SSDLoss(num_classes=6)

net = torch.nn.DataParallel(net, device_ids=[0,1])
cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * end_epoch), int(0.75 * end_epoch)], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets,_,__) in enumerate(trainloader):
        images = images.to(device)
        loc_targets = loc_targets.to(device)
        conf_targets = conf_targets.to(device)

        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        # correct += compute_class_acc(conf_preds,conf_targets)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        # print('Total loss: %.3f | Average loss: %.3f ' % (loss.data.item(),train_loss/(batch_idx+1) ) )
    print('\nAverage loss/batch:  %.3f \n' %(train_loss/(batch_idx+1) ) )

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    total = 0
    # for decode: nms
    data_encoder = DataEncoder()
    acc,mean_iou = 0,0
    with torch.no_grad():
        for batch_idx, (images, loc_targets, conf_targets,real_boxes,real_class) in enumerate(testloader):
            images = images.to(device)
            loc_targets ,conf_targets= loc_targets.to(device) ,conf_targets.to(device)
            real_boxes ,real_class = real_boxes.squeeze().to(device) ,real_class.squeeze().to(device)
            # print(loc_targets.size(),conf_targets.size(),real_boxes.size(),real_class.size()) #(30, 8732, 4) (30, 8732) (30, 4) (30,)
            loc_preds, conf_preds = net(images)
            # print(loc_preds.size(),conf_preds.size(),loc_preds[0].size(),F.softmax(conf_preds[0]).size())#(30,8732,4)(30,8732, 6) (8732, 4) (8732, 6)
            loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
            test_loss += loss.data.item()
            total += images.size(0)
            # compute acc and mean_iou
            fin_bboxes = torch.zeros(batch_size,4) # [N,4]
            fin_class = torch.zeros(batch_size,) # [N,]
            for i in range(batch_size):
                # loc_preds: (N,8732, 4)
                # conf_preds: (N, 8732, 6)
                # real_boxes: (N,4)
                # real_class: (N,)
                # boxes: (tensor) bbox locations, sized [#obj, 4].
                # labels: (tensor) class labels, sized [#obj,1].
                # return ([tensor([[0.2015, 0.2993, 0.8239, 0.6976]])], [tensor([2])], [tensor([0.9855])])
                boxes,labels,score = data_encoder.decode(loc_preds[i].data.cpu(), F.softmax(conf_preds[i],dim=1).data.cpu(),threshold=0.1)#
                fin_bboxes[i], fin_class[i]= boxes,labels
            # print('\n class:',fin_class.to(device).long(),real_class+1,'\n bbox:',fin_bboxes.to(device),real_boxes)
            class_acc,IoU = compute_iou_acc(fin_class.to(device).long(),real_class+1,fin_bboxes.to(device),real_boxes)
            acc += class_acc
            mean_iou += IoU
        print('\nAverage loss/batch: %.3f Mean IoU: %.3f Classificatin Acc: %.3f' %(test_loss/(batch_idx+1),mean_iou/total,1.*acc/total) )

    # Save checkpoint.
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ssd300_ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+end_epoch):
    train(epoch)
    test(epoch)
    scheduler.step