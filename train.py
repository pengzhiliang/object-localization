#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

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
from torchvision import datasets, models, transforms
from torchsummary import summary
from model import Net
from metrics import averageMeter,compute_iou_acc
from dataset import tiny_vid_loader,xywh_to_x1y1x2y2, x1y1x2y2_to_xywh

# 参数设置
defualt_path = '/home/pzl/Data/tiny_vid'
learning_rate = 1e-7
batch_size = 50
num_workers = 12
resume_path = '/home/pzl/object-localization/checkpoint/best_model.pkl'
resume_flag = True
start_epoch = 0
end_epoch = 1000
test_interval = 10
print_interval = 1

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Dataloader
train_loader = tiny_vid_loader(defualt_path=defualt_path,mode='train')
test_loader = tiny_vid_loader(defualt_path=defualt_path,mode='test',transform = None)
trainloader = DataLoader(train_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
testloader = DataLoader(test_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

# Setup Model and summary
model = Net('mobilenet',freeze_basenet = False).to(device)
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
summary(model,(3,128,128)) # summary 网络参数

# 需要学习的参数
# base_learning_list = list(filter(lambda p: p.requires_grad, model.base_net.parameters()))
# learning_list = model.parameters()

# 优化器以及学习率设置
optimizer = SGD([
					{'params': model.base_net.parameters(),'lr': learning_rate / 10},
					{'params': model.model_class.parameters(), 'lr': learning_rate * 10},
					{'params': model.model_reg.parameters(), 'lr': learning_rate * 10}
				], lr=learning_rate, momentum=0.99, weight_decay=5e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.4 * end_epoch), int(0.7 * end_epoch),int(0.8 * end_epoch),int(0.9 * end_epoch)], gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=10, verbose=True)
loss_class = nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)
loss_reg = nn.SmoothL1Loss(reduction='sum').to(device) # or MSELoss or L1Loss or SmoothL1Loss

# resume 
if (os.path.isfile(resume_path) and resume_flag):
	checkpoint = torch.load(resume_path)
	model.load_state_dict(checkpoint["model_state"])
	optimizer.load_state_dict(checkpoint["optimizer_state"])
	# scheduler.load_state_dict(checkpoint["scheduler_state"])
	# start_epoch = checkpoint["epoch"]
	print("=====>",
	    "Loaded checkpoint '{}' (iter {})".format(
	        resume_path, checkpoint["epoch"]
	    )
	)
else:
    print("=====>","No checkpoint found at '{}'".format(resume_path))

# Training
def train(epoch,display=True):
	print('\nEpoch: %d' % epoch)
	model.train()
	train_class_loss = 0
	train_reg_loss = 0
	correct = 0
	IoU = 0.
	total = 0
	for batch_idx, (inputs, targets_class,targets_reg) in enumerate(trainloader):
		inputs, targets_class,targets_reg = inputs.to(device), targets_class.to(device),targets_reg.to(device).float()
		optimizer.zero_grad()
		# 预测的为物体中心及宽高
		outputs_reg,outputs_class = model(inputs)
		targets_reg = x1y1x2y2_to_xywh(targets_reg/128.)
		c_loss = loss_class(outputs_class, targets_class)
		# print('out:',targets_reg.data)
		r_loss = loss_reg(outputs_reg, targets_reg)
		loss = c_loss + r_loss
		loss.backward()
		optimizer.step()

		train_class_loss += c_loss.item()
		train_reg_loss += r_loss.item()

		class_acc,batch_IoU = compute_iou_acc(outputs_class,targets_class,xywh_to_x1y1x2y2(outputs_reg),xywh_to_x1y1x2y2(targets_reg))
		total += targets_reg.size(0)
		correct += class_acc
		IoU += batch_IoU
		if display:
			# progress_bar(batch_idx, len(trainloader), 'Total Loss: %.4f|C_Loss: %.4f|R_loss: %.4f|M_Iou :%.4f |Acc: %.4f%% (%d/%d)'
			# 		 % ((train_class_loss+train_reg_loss)/(batch_idx+1),train_class_loss/(batch_idx+1),train_reg_loss/(batch_idx+1),IoU/total, 100.*correct/total, correct, total))
			print('Total Loss: %.4f|C_Loss: %.4f|R_loss: %.4f|M_Iou :%.4f |Acc: %.4f%% (%d/%d)'
				% ((train_class_loss+train_reg_loss)/(batch_idx+1),train_class_loss/(batch_idx+1),train_reg_loss/(batch_idx+1),IoU/total, 100.*correct/total, correct, total))
	# print('\n Output: \n',xywh_to_x1y1x2y2(outputs_reg.data*128)[0])
	# print('\n target: \n',xywh_to_x1y1x2y2(targets_reg.data*128)[0])
	return (train_class_loss+train_reg_loss)/(batch_idx+1),1.*correct/total

def test(epoch,display=True):
	global best_acc
	model.eval()
	print('Test')
	test_class_loss = 0
	test_reg_loss = 0
	correct = 0
	IoU = 0.
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets_class,targets_reg) in enumerate(testloader):
			inputs, targets_class,targets_reg = inputs.to(device), targets_class.to(device),targets_reg.to(device).float()
			optimizer.zero_grad()
			outputs_reg,outputs_class = model(inputs)

			targets_reg = x1y1x2y2_to_xywh(targets_reg/128.)

			c_loss = loss_class(outputs_class, targets_class)
			r_loss = loss_reg(outputs_reg, targets_reg)
			loss = c_loss + r_loss

			test_class_loss += c_loss.item()
			test_reg_loss += r_loss.item()

			# _, predicted = outputs_class.max(dim=1)
			class_acc,batch_IoU = compute_iou_acc(outputs_class,targets_class,xywh_to_x1y1x2y2(outputs_reg),xywh_to_x1y1x2y2(targets_reg))
			total += targets_reg.size(0)
			correct += class_acc
			IoU += batch_IoU
		if display:
				# progress_bar(batch_idx, len(trainloader), 'Test Loss: %.4f| C_Loss: %.4f|R_loss: %.4f|M_Iou :%.4f|Acc: %.4f%% (%d/%d)'
				# 		 % ((test_class_loss+test_reg_loss)/(batch_idx+1),test_class_loss/(batch_idx+1),test_reg_loss/(batch_idx+1),IoU/total, 100.*correct/total, correct, total))
				print('Test Loss: %.4f| C_Loss: %.4f|R_loss: %.4f|M_Iou :%.4f|Acc: %.4f%% (%d/%d)'
					% ((test_class_loss+test_reg_loss)/(batch_idx+1),test_class_loss/(batch_idx+1),test_reg_loss/(batch_idx+1),IoU/total, 100.*correct/total, correct, total))
		if 1.*correct/total > best_acc:
			best_acc = 1.*correct/total
			state = {
			    "epoch": epoch + 1,
			    "model_state": model.state_dict(),
			    "optimizer_state": optimizer.state_dict(),
			    "scheduler_state": scheduler.state_dict(),
			    "best_acc": best_acc,
			}
			save_path = os.path.join(os.path.split(resume_path)[0],"best_model.pkl")
			print("saving......")
			torch.save(state, save_path)
	return (test_class_loss+test_reg_loss)/(batch_idx+1),1.*correct/total

train_acc = []
train_loss = []
test_acc = []
test_loss = []
display = True
best_acc = 0.
# if display:
#     from utils import progress_bar
for epoch in range(start_epoch, end_epoch):
	loss,acc = train(epoch,display)
	train_loss.append(loss)
	train_acc.append(acc)

	if (epoch+1) % test_interval == 0 or epoch+1 == end_epoch:
		loss,acc = test(epoch,display)
		test_loss.append(loss)
		test_acc.append(acc)
		scheduler.step(loss)
    # print(train_loss[-1],train_acc[-1],test_loss[-1],test_acc[-1])








