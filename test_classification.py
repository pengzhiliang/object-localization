#-*-coding:utf-8-*-
'''
Created on Oct 31,2018
#
#
#
#		TEST for classification using CIFAR10 datasets
#
#
#
#
@author: pengzhiliang
'''
from __future__ import print_function

import time 
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler,Adam,SGD
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchsummary import summary
from model import MobileNet
from metrics import compute_acc,averageMeter,compute_IoU,compute_class_acc
from dataset import tiny_vid_loader

def get_mobilenet_model(pretain = True,num_classes = 5,requires_grad = True):
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
			torch.nn.Linear(128, 10)
		)
	def forward(self,x):
		out = self.base_net(x) 					#out: (n,1024,4,4)
		# out = F.avg_pool2d(out, 4) 				#out: (n,1024,1,1)
		out = out.view(-1,self.in_features) 	#out: (n,1024)
		# out_reg = self.model_reg(out) 				# out: (n,4)
		out_class = self.model_class(out)			# out: (n,5)

		return out_class

# 参数设置
defualt_path = '/home/pzl/Data/tiny_vid'
learning_rate = 1e-3
batch_size = 32
num_workers = 4
start_iter = 0
end_iter = 1000
test_interval = 10
print_interval = 1

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Dataloader
# train_loader = tiny_vid_loader(defualt_path=defualt_path,mode='train')
# test_loader = tiny_vid_loader(defualt_path=defualt_path,mode='test')
# trainloader = DataLoader(train_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
# testloader = DataLoader(test_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

# Test CIFAR10 Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root="/home/pzl/CV/cifar10/", train=True, download=False, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = datasets.CIFAR10(root="/home/pzl/CV/cifar10/", train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Setup Model and summary
model = Net().to(device)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
# summary(model,(3,32,32)) # summary 网络参数

# 参数学习列表
# learning_list = list(filter(lambda p: p.requires_grad, model.parameters()))
learning_list = model.parameters()
# 优化器以及学习率设置
optimizer = SGD(learning_list, lr=learning_rate)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * end_iter), int(0.75 * end_iter)], gamma=0.1)
# loss_class = nn.CrossEntropyLoss()#reduction='elementwise_mean').to(device)


flag = True
best_acc = 0.0
i = start_iter
# 记录器
loss_meter = averageMeter()
time_meter = averageMeter()
class_acc_meter = averageMeter()

while i <= end_iter and flag:
	for (images, gt) in trainloader:
		i += 1

		start_ts = time.time()	
		model.train()
		images,gt = images.to(device),gt.to(device)
		# gt_class = gt[:,:1].long()
		# 优化器置0
		optimizer.zero_grad()
		out_class = model(images)
		loss = F.cross_entropy(out_class, gt)

		loss.backward()
		optimizer.step()
		# scheduler.step()

		time_meter.update(time.time() - start_ts)
		# 每print_interval显示一次
		if (i + 1) % print_interval == 0:
			fmt_str = "Iter [{:d}/{:d}]  classification loss:{:.4f}  Time/Image:{:.4f}"
			print_str = fmt_str.format(i + 1,end_iter,loss.item(),time_meter.avg / batch_size)
			print(print_str)
			time_meter.reset()
		# 每test_interval test一次
		if (i + 1) % test_interval == 0 or (i + 1) == end_iter:
			model.eval()
			with torch.no_grad():
				for i_te, (images_te, gt_te) in tqdm(enumerate(testloader)):
					images_te,gt_te  = images_te.to(device),gt_te.to(device)
					# gt_class = gt[:,:1].long()

					out_class = model(images_te)
					# loss = loss_class(out_class,gt_te)
					loss = F.cross_entropy(out_class, gt_te)
					# loss = loss_class(out_class,gt_class.squeeze(1))
					class_acc = compute_class_acc(out_class,gt_te)/float(batch_size)
					# class_acc = compute_class_acc(out_class,gt_class.squeeze(1))
					# class_acc,mean_IoU = compute_acc(clas,coor,gt_te)
					class_acc_meter.update(class_acc)
					loss_meter.update(loss.item())
			print("Iter %d Loss: %.4f classification acc:%.4f" % (i + 1,loss_meter.avg,class_acc_meter.avg))
			class_acc_meter.reset()
			loss_meter.reset()

		if (i + 1) == end_iter:
			flag = False
			break










