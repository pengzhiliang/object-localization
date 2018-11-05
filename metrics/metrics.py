#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

@author: pengzhiliang
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_IoU(boxes1, boxes2):
	"""
	计算IOU
	参数：
		boxes1: gt (n,4)
		boxes2: pred (n,4)
	返回：
		（n,）
	"""
	xA = torch.max(boxes1[:, 0], boxes2[:, 0])
	yA = torch.max(boxes1[:, 1], boxes2[:, 1])
	xB = torch.min(boxes1[:, 2], boxes2[:, 2])
	yB = torch.min(boxes1[:, 3], boxes2[:, 3])
	interArea = (xB - xA + 1) * (yB - yA + 1)
	boxAArea = (boxes1[:,2] - boxes1[:,0] + 1) * (boxes1[:,3] - boxes1[:,1] + 1)
	boxBArea = (boxes2[:,2] - boxes2[:,0] + 1) * (boxes2[:,3] - boxes2[:,1] + 1)
	iou = interArea / (boxAArea + boxBArea - interArea)
	iou[iou<0] = 0
	return iou


def compute_class_acc(in_class,gt_class):
	# 单独计算分类结果，并返回正确个数
	_,pred_class = in_class.max(dim =1) # pred_class :(n,),取值0~4
	class_acc = pred_class.eq(gt_class).sum().item()
	return class_acc

def compute_iou_acc(in_class,gt_class,in_coor,gt_coor,theta=0.5):
	"""
	设置阈值为0.5  <0.5不将其算入分类准确率
	参数：
		in_class: 预测的class分类,(n,)
		in_coor : 预测坐标, (n,4)
		gt: groundtruth (n,5)
	返回：
		分类准确个数，IoU
	"""
	in_coor[in_coor<0] = 0
	in_coor[in_coor>1] = 1
	IoU = compute_IoU(gt_coor*128,in_coor*128) # (n,)
	mean_IoU = IoU.sum().item()
	if len(in_class.size()) == 1:
		# background class = 0
		# print(in_class,IoU,gt_class)
		in_class[IoU<theta] = 100
		class_acc = in_class.eq(gt_class).sum().item()
	else:
		_,pred_class = in_class.max(dim =1) # pred_class :(n,),取值0~4
		# TODO: <0.5不将其算入分类准确率
		pred_class[IoU<0.5] = 100
		class_acc = pred_class.eq(gt_class).sum().item()
	print(class_acc,mean_IoU)
	return class_acc,mean_IoU

class averageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count