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
	return iou
    # intersec = boxes1.clone()
    # intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    # intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    # intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    # intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    # def compute_area(boxes):
    #     # in (x1, y1, x2, y2) format
    #     dx = boxes[:, 2] - boxes[:, 0]
    #     dx[dx < 0] = 0
    #     dy = boxes[:, 3] - boxes[:, 1]
    #     dy[dy < 0] = 0
    #     return dx * dy
    
    # a1 = compute_area(boxes1)
    # a2 = compute_area(boxes2)
    # ia = compute_area(intersec)
    # assert((a1 + a2 - ia <= 0).sum() == 0)
    
    # return ia / (a1 + a2 - ia) 

def compute_class_acc(in_class,gt_class):
	_,pred_class = in_class.max(dim =1) # pred_class :(n,),取值0~4
	class_acc = pred_class.eq(gt_class).sum().item()
	return class_acc

def compute_acc(in_class,in_coor,gt, theta=0.5):
	"""
	设置阈值为0.5  <0.5不将其算入分类准确率
	参数：
		in_class: 预测的class分类,(n,5)
		in_coor : 预测坐标, (n,4)
		gt: groundtruth (n,5)
	返回：
		分类准确率，mean IoU
	"""
	if not isinstance(gt,torch.Tensor):
		gt = torch.tensor(gt)
	gt_class = gt[:,:1] # 真实分类 (n,1)
	gt_coor = gt[:,1:] #　真实坐标 (n,4)
	in_coor = (in_coor*128).long()
	num = gt.size(0)

	_,pred_class = in_class.max(dim =1) # pred_class :(n,),取值0~4
	IoU = compute_IoU(gt_coor,in_coor) # (n,)
	# print('\n',gt_coor,'\n',in_coor,'\n',IoU)
	# TODO: <0.5不将其算入分类准确率
	# pred_class[IoU<0.5] = 5
	class_acc = pred_class.eq(gt_class.squeeze(1)).sum().item()/float(num)

	mean_IoU = IoU.sum()/float(num)

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