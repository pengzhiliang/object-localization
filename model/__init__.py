#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

@author: pengzhiliang
'''

from mobilenet import *
from vgg import *
from ssd import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mobilenet_model(pretain = True,num_classes = 5,requires_grad = False):
	# 返回去掉了全连接层的mobilenet
	model = MobileNet()
	# 不训练这几层
	for param in model.parameters():
		param.requires_grad = requires_grad

	if pretain:
		# Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
		basenet_state = torch.load("./pretrained/mobienetv2.pth")
		# filter out unnecessary keys
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
		# load the new state dict
		model.load_state_dict(pretrained_dict)
		return model
	else:
		return model
def get_vgg_model(vggname='VGG16',pretain = True,num_classes = 5,requires_grad = False):
	# 返回去掉了全连接层的mobilenet
	model = VGG(vggname) # (n,512,4,4)
	# 不训练这几层
	for param in model.parameters():
		param.requires_grad = requires_grad

	if pretain:
		# Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
		basenet_state = torch.load("/home/pzl/CV/2/vgg16_bn-6c64b313.pth")
		# filter out unnecessary keys
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
		# load the new state dict
		model.load_state_dict(pretrained_dict)
		return model
	else:
		return model

class Net(nn.Module):
	def __init__(self,netname,freeze_basenet = False):
		super(Net,self).__init__()
		if netname[:3] == 'VGG':
			self.base_net = get_vgg_model(netname,requires_grad = not freeze_basenet)
			self.in_features = 512
		else:
			self.base_net = get_mobilenet_model(requires_grad = not freeze_basenet)
			self.in_features = 1024

		# 预测四个坐标
		self.model_reg = torch.nn.Sequential(
			torch.nn.Linear(self.in_features, 256),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(128, 4)
		)
		# 预测分类
		self.model_class = torch.nn.Sequential(
			torch.nn.Linear(self.in_features, 1024),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(1024,128),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(128, 5)
		)
	def forward(self,x):
		out = self.base_net(x) 					#out: (n,in_features,4,4)
		out = F.avg_pool2d(out, 4) 				#out: (n,in_features,1,1)
		out = out.view(-1,self.in_features) 	#out: (n,in_features)
		out_reg = self.model_reg(out) 				# out: (n,4)
		out_class = self.model_class(out)			# out: (n,5)

		return out_reg,out_class


