#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
'''
from __future__ import print_function
import torch
import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from os.path import join as pjoin

def train_test_txt(defualt_path='/home/pzl/Data/tiny_vid'):
	"""
	将如下格式存入文件：
		/home/pzl/Data/tiny_vid/turtle/000151.JPEG	2	29 38 108 84
	其中参数：
		/home/pzl/Data/tiny_vid/turtle/000151.JPEG：图片路径
		2 ： 种类
		29 38 108 84：分别为 xmin, ymin, xmax, ymax，表示bounding box的位置
	"""
	classes = {'car':0, 'bird':1, 'turtle':2, 'dog':3, 'lizard':4}
	for dirname in classes.keys():
		bbox_dic = {}
		with open(pjoin(defualt_path,dirname+'_gt.txt'),'r') as f:
			for n,line in enumerate(f.readlines()):
				line = line.strip().split()
				bbox_dic[line[0]] = line[1:]
				if n == 179:
					break
		with open(pjoin(defualt_path,'train_images.txt'),'a') as f:
			for i in range(1,151): 
				imgname = '000000'
				pad0 = 6 - len(str(i))
				imgname = imgname[:pad0]+str(i)+'.JPEG'
				imgpath = pjoin(pjoin(defualt_path,dirname),imgname)
				imageclass = str(classes[dirname])
				imgbbox = ' '.join(bbox_dic[str(i)])
				f.write('\t'.join([imgpath,imageclass,imgbbox])+'\n')
		with open(pjoin(defualt_path,'test_images.txt'),'a') as f:
			for i in range(151,181): 
				imgname = '000000'
				pad0 = 6 - len(str(i))
				imgname = imgname[:pad0]+str(i)+'.JPEG'
				imgpath = pjoin(pjoin(defualt_path,dirname),imgname)
				imageclass = str(classes[dirname])
				imgbbox = ' '.join(bbox_dic[str(i)])
				f.write('\t'.join([imgpath,imageclass,imgbbox])+'\n')


class tiny_vid_loader(data.Dataset):
	"""
	功能：
		构造一个用于tiny_vid数据集的迭代器
	参数：

	"""
	def __init__(self,defualt_path='/home/pzl/Data/tiny_vid',mode='train',transform=None):
		"""
		defualt_path: 如'/home/pzl/Data/tiny_vid'
		mode : 'train' or 'test'
		"""
		if not (os.path.exists(pjoin(defualt_path,'train_images.txt')) and os.path.exists(pjoin(defualt_path,'test_images.txt'))):
			train_test_txt(defualt_path)
		self.filelist=[]
		self.class_coor = []
		with open(pjoin(defualt_path,mode+'_images.txt')) as f:
			for line in f.readlines():
				line = line.strip().split()
				self.filelist.append(line[0])
				self.class_coor.append([int(i) for i in line[1:]])
		if transform is None:
			self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		else:
			self.transform = transform

	def __getitem__(self, index):
		imgpath = self.filelist[index]
		img = Image.open(imgpath).convert('RGB')
		img = self.transform(img)
		class_gt = np.array(self.class_coor[index])/128.
		class_gt = torch.tensor(class_gt)
		return img,class_gt

	def __len__(self):
		return len(self.filelist)

if __name__ == '__main__':
	target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
	dst = tiny_vid_loader()
	trainloader = data.DataLoader(dst, batch_size=1)
	for i, data in enumerate(trainloader):
		if i == 224:
			img, class_gt = data
			class_gt=class_gt.numpy()
			print(class_gt)

			inp = img.numpy()[0].transpose((1, 2, 0))
			mean = np.array([0.485, 0.456, 0.406])
			std = np.array([0.229, 0.224, 0.225])
			inp = std * inp + mean
			inp = np.clip(inp, 0, 1)*255
			inp = Image.fromarray(inp.astype('uint8')).convert('RGB')
			dis_gt(inp,target_classes[class_gt[0,0]],class_gt[0,1:])

			break
    