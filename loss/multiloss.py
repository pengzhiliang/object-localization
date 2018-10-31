#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
'''
import torch.nn as nn
import torch.nn.functional as F
import torch


def multi_loss(in_class,in_coor,gt):
	"""
	计算分类和回归的总loss
	参数：
		in_class: 输入的预测种类	(n,5)
		in_coor: 输入的预测坐标	(n,4)
		gt : 真实种类以及坐标		(n,5) 第一个为种类，后四个为坐标
	"""
	if not isinstance(gt,torch.Tensor):
		gt = torch.tensor(gt)
	gt_class = gt[:,:1]
	gt_coor = gt[:,1:]
	# 预测坐标回归损失
	regress_loss = F.smooth_l1_loss(in_coor, gt_coor, reduction='elementwise_mean')
	# 分类损失
	classification_loss = F.cross_entropy(in_class, gt_class.squeeze(1),reduction='elementwise_mean')
	# 返回
	return regress_loss,classification_loss,regress_loss+classification_loss
