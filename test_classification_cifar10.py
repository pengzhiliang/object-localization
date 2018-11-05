#-*-coding:utf-8-*-

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler,Adam,SGD
from model import MobileNet
from dataset import tiny_vid_loader

import os
import numpy as np


def get_mobilenet_model(pretain = True,num_classes = 5,requires_grad = False):
    # 返回去掉了全连接层的mobilenet
    model = MobileNet()
    # 不训练这几层
    for param in model.parameters():
        param.requires_grad = requires_grad

    if pretain:
        # Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
        basenet_state = torch.load("/home/pzl/object-localization/pretained/mobienetv2.pth")
        # filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
        # load the new state dict
        model.load_state_dict(pretrained_dict)
        return model
    else:
        return model

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.base_net = get_mobilenet_model()
        # self.base_net = MobileNet()
        self.in_features = 1024
        # 预测四个坐标
        self.model_reg = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 4)
        )
        # 预测分类
        self.model_class = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512,128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 5)
        )
    def forward(self,x):
        out = self.base_net(x)                  #out: (n,1024,4,4)
        out = F.avg_pool2d(out, 4)                #out: (n,1024,1,1)
        out = out.view(-1,self.in_features)     #out: (n,1024)
        # out_reg = self.model_reg(out)                 # out: (n,4)
        out_class = self.model_class(out)           # out: (n,5)

        return out_class

defualt_path = '/home/pzl/Data/tiny_vid'
learning_rate = 1e-3
batch_size = 32
num_workers = 4
start_iter = 0
end_iter = 1000
test_interval = 10
print_interval = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# # Test CIFAR10 Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = datasets.CIFAR10(root="/home/pzl/CV/cifar10/", train=True, download=False, transform=transform_train)
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# testset = datasets.CIFAR10(root="/home/pzl/CV/cifar10/", train=False, download=False, transform=transform_test)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# #TEST tiny_vid dataset
print('==> Preparing data..')
train_loader = tiny_vid_loader(defualt_path=defualt_path,mode='train')
test_loader = tiny_vid_loader(defualt_path=defualt_path,mode='test')
trainloader = DataLoader(train_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
testloader = DataLoader(test_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

# Model
print('==> Building model..')
# net = VGG('VGG16')
net = Net()
print(net)

net = net.to(device)

criterion = nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)
optimizer = SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch,display=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        targets = (targets[:,:1].long()).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if display:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1),1.*correct/total

def test(epoch,display=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets = (targets[:,:1].long()).squeeze(1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if display:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return test_loss/(batch_idx+1),acc/100.

### adjust_learning_rate
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs: lr = args.lr * (0.1 ** (epoch // 30))""" 
    lr = learning_rate * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


train_acc = []
train_loss = []
test_acc = []
test_loss = []
display = True
if display:
    from utils import progress_bar
for epoch in range(start_epoch, start_epoch+300):
    loss,acc = train(epoch,display)
    train_loss.append(loss)
    train_acc.append(acc)

    loss,acc = test(epoch,display)
    test_loss.append(loss)
    test_acc.append(acc)
    print(train_loss[-1],train_acc[-1],test_loss[-1],test_acc[-1])

    adjust_learning_rate(optimizer, epoch)
