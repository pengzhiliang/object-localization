#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

@author: pengzhiliang
'''
from __future__ import print_function

import time 
import tqdm
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.optim import lr_scheduler,Adam,SGD
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchsummary import summary
from model import Net
from loss import multi_loss
from metrics import compute_acc,averageMeter
from dataset import tiny_vid_loader

# 参数设置
defualt_path = '/home/pzl/Data/tiny_vid'
learning_rate = 1e-5
batch_size = 32
num_workers = 12
resume_path = 'checkpoint/best_model.pkl'
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
model = Net().to(device)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
summary(model,(3,128,128)) # summary 网络参数

# 优化器以及学习率设置
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# resume 
if os.path.isfile(resume_path):
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
best_iou = -100.0
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
		optimizer.zero_grad()
		# 多输出
		clas,coor = model(images)
		# 损失
		r_loss,c_loos,loss = multi_loss(clas,coor,gt)
		# loss bp
		loss.backward()
		optimizer.step()

		time_meter.update(time.time() - start_ts)
		# 每print_interval显示一次
		if (i + 1) % print_interval == 0:
			fmt_str = "Iter [{:d}/{:d}]  regress loss:{:.4f},classification loss:{:.4f} Total Loss:{:.4f}  Time/Image:{:.4f}"
			print_str = fmt_str.format(i + 1,end_iter,r_loss.item(),c_loos.item(),loss.item(),time_meter.avg / batch_size)
			print(print_str)
			time_meter.reset()
		# 每test_interval test一次
		if (i + 1) % test_interval == 0 or (i + 1) == end_iter:
			model.eval()
			with torch.no_grad():
				for i_te, (images_te, gt_te) in tqdm(enumerate(valloader)):
					images_te = images_te.to(device)

					clas,coor = model(images_te)
					r_loss,c_loos,loss = multi_loss(clas,coor,gt_te)

					class_acc,mean_IoU = compute_acc(clas.data.cpu(),coor.data.cpu(),gt_te)

					class_acc_meter.update(class_acc)
					mean_IoU_meter.update(mean_IoU)
					loss_meter.update(loss.item())

			print("Iter %d Loss: %.4f classification acc:%.4f Mean Iou:%.4f" % (i + 1,loss_meter.avg,class_acc_meter.avg,mean_IoU_meter.avg))



			if mean_IoU_meter.avg > best_iou:
				best_iou = mean_IoU_meter.avg
				state = {
				    "epoch": i + 1,
				    "model_state": model.state_dict(),
				    "optimizer_state": optimizer.state_dict(),
				    "scheduler_state": scheduler.state_dict(),
				    "best_iou": best_iou,
				}
				save_path = os.path.join('./checkpoint',"{}_best_model.pkl".format(i))
				print("saving......")
				torch.save(state, save_path)

			class_acc_meter.reset()
			mean_IoU_meter.reset()
			loss_meter.reset()

		if (i + 1) == end_iter:
			flag = False
			break










