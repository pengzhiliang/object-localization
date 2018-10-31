#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

@author: pengzhiliang
'''

from mobilenet import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mobilenet_model(pretain = True,num_classes = 5):
	# 返回去掉了全连接层的mobilenet
	model = MobileNet()
	if pretain:
		# Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
		basenet_state = torch.load('pretrained/mobienetv2.pth')
		# filter out unnecessary keys
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
		# load the new state dict
		model.load_state_dict(pretrained_dict)
		return model
	else:
		return model

def forward_from(module_seq, start_idx, end_index, input_x):
    """
    Forward the network from layer
    :param module_seq: a sequential of network layers, must be nn.Sequential
    :param start_idx: start index of
    :param end_index: end index of forwarding layer
    :param input_x: input tensor to be forwarded
    :return: result of forwarding multiple layers
    """
    x = input_x
    for layer in module_seq[start_idx: end_index]:
        x = layer(x)
    return x


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.base_net = get_mobilenet_model()
		# self.base_net = MobileNet()
		self.in_features = 1024
		# 预测四个坐标
		self.regressor = nn.Sequential(
				nn.Linear(self.in_features,512),
				nn.ReLU(True),
				nn.Dropout(), 
				nn.Linear(512,4)
			)
		# 预测分类
		self.classifier = nn.Sequential(
				nn.Linear(self.in_features,512),
				nn.ReLU(True),
				nn.Dropout(), 
				nn.Linear(512,5)
			)

	def forward(self,x):
		out = self.base_net(x) 					#out: (n,1024,4,4)
		out = F.avg_pool2d(out, 4) 				#out: (n,1024,1,1)
		out = out.view(-1,self.in_features) 	#out: (n,1024)
		coor = self.regressor(out) 				# out: (n,4)
		clas = self.classifier(out)				# out: (n,5)

		return clas,coor


