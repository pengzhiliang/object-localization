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
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchsummary import summary
from model import Net
from loss import multi_loss
from metrics import compute_acc,averageMeter,compute_IoU,compute_class_acc
from dataset import tiny_vid_loader

# 参数设置
defualt_path = '/home/pzl/Data/tiny_vid'
learning_rate = 1e-5
batch_size = 50
num_workers = 12
resume_path = '/home/pzl/object-localization/checkpoint/best_model.pkl'
resume_flag = False
start_iter = 0
end_iter = 1000
test_interval = 5
print_interval = 1

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Dataloader
train_loader = tiny_vid_loader(defualt_path=defualt_path,mode='train')
test_loader = tiny_vid_loader(defualt_path=defualt_path,mode='test')
trainloader = DataLoader(train_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
testloader = DataLoader(test_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

# Setup Model and summary
model = Net('VGG16').to(device)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
summary(model,(3,128,128)) # summary 网络参数

learning_list = list(filter(lambda p: p.requires_grad, model.parameters()))
# 优化器以及学习率设置
optimizer = SGD(learning_list, lr=learning_rate)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * end_iter), int(0.75 * end_iter)], gamma=0.1)
loss_class = nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)
loss_reg = nn.MSELoss(reduction='elementwise_mean').to(device)

# resume 
if (os.path.isfile(resume_path) and resume_flag):
	checkpoint = torch.load(resume_path)
	model.load_state_dict(checkpoint["model_state"])
	optimizer.load_state_dict(checkpoint["optimizer_state"])
	scheduler.load_state_dict(checkpoint["scheduler_state"])
	start_iter = checkpoint["epoch"]
	print("=====>",
	    "Loaded checkpoint '{}' (iter {})".format(
	        resume_path, checkpoint["epoch"]
	    )
	)
else:
    print("=====>","No checkpoint found at '{}'".format(resume_path))

flag = True
best_acc = 0.0
i = start_iter

loss_meter = averageMeter()
time_meter = averageMeter()
class_acc_meter = averageMeter()
mean_IoU_meter = averageMeter()

while i <= end_iter and flag:
	for (images, gt) in trainloader:
		i += 1

		start_ts = time.time()
		scheduler.step()
		model.train()
		images = images.to(device)
		gt = gt.to(device)
		gt_class = gt[:,:1].long()
		gt_reg = (gt[:,1:]/128.).float()
		# 优化器置0
		optimizer.zero_grad()
		# 多输出
		out_reg,out_class = model(images)
		# r_loss,c_loss,loss = multi_loss(clas,coor,gt)
		r_loss = loss_reg(out_reg,gt_reg)#, reduction='elementwise_mean')
		c_loss = loss_class(out_class,gt_class.squeeze(1))#,reduction='elementwise_mean')
		loss = r_loss+c_loss
		# for test
		# print(out_class.data.max(dim=1)[1])
		# print(gt_class.squeeze(1))
		# print(c_loss.data)
		# loss bp
		loss.backward()
		optimizer.step()

		time_meter.update(time.time() - start_ts)
		# 每print_interval显示一次
		if (i + 1) % print_interval == 0:
			fmt_str = "Iter [{:d}/{:d}]  regress loss:{:.4f},classification loss:{:.4f} Total Loss:{:.4f}  Time/Image:{:.4f}"
			print_str = fmt_str.format(i + 1,end_iter,r_loss.item(),c_loss.item(),loss.item(),time_meter.avg / batch_size)
			print(print_str)
			time_meter.reset()
		# 每test_interval test一次
		if (i + 1) % test_interval == 0 or (i + 1) == end_iter:
			model.eval()
			with torch.no_grad():
				for i_te, (images_te, gt_te) in tqdm(enumerate(testloader)):
					images_te = images_te.to(device)
					gt_te = gt_te.to(device)
					gt_class = gt[:,:1].long()
					gt_reg = (gt[:,1:]/128.).float()

					out_reg,out_class = model(images_te)
					r_loss = loss_reg(out_reg,gt_reg)#, reduction='elementwise_mean')
					c_loss = loss_class(out_class,gt_class.squeeze(1))#,reduction='elementwise_mean')
					loss = r_loss+c_loss

					class_acc = compute_class_acc(out_class,gt_class.squeeze(1))
					IoU = compute_IoU(out_reg,gt_reg)
					mean_IoU = IoU.sum()/batch_size
					class_acc = class_acc/float(batch_size)
					# class_acc,mean_IoU = compute_acc(clas,coor,gt_te)

					class_acc_meter.update(class_acc)
					mean_IoU_meter.update(mean_IoU)
					loss_meter.update(loss.item())
			# print(class_acc,IoU)
			print("Iter %d Loss: %.4f classification acc:%.4f Mean Iou:%.4f" % (i + 1,loss_meter.avg,class_acc_meter.avg,mean_IoU_meter.avg))



			if class_acc_meter.avg > best_acc:
				best_acc = class_acc_meter.avg
				state = {
				    "epoch": i + 1,
				    "model_state": model.state_dict(),
				    "optimizer_state": optimizer.state_dict(),
				    "scheduler_state": scheduler.state_dict(),
				    "best_acc": best_acc,
				}
				save_path = os.path.join(os.path.split(resume_path)[0],"best_model.pkl")
				print("saving......")
				torch.save(state, save_path)

			class_acc_meter.reset()
			mean_IoU_meter.reset()
			loss_meter.reset()

		if (i + 1) == end_iter:
			flag = False
			break










