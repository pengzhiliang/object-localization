#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
@version: 1.1
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageFont,ImageDraw

def dis_gt(img ,name , coor ):
	"""
	功能：在输入的Image上加上bounding box
	参数：
		img： 输入图像
		name: Object class
		cor: np.array一维数组，四个坐标，依次为（xmin, ymin, xmax, ymax）
	返回：
		显示图片
	"""
	# # 利用opencv 画出矩形
	# cv2.rectangle(img,(coor[0],coor[1]),(coor[2],coor[3]),(0,255,0),1)
	# # 在ground truth上加上class name
	# font = cv2.FONT_HERSHEY_SIMPLEX
	# # 参数： 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
	# cv2.putText(img,name,(coor[0],coor[1]+15), font, 0.5,(0,255,0),2,cv2.LINE_AA)

	# plt.imshow(img)
	# plt.show()
	draw=ImageDraw.Draw(img)
	# newfont=ImageFont.truetype('simkai.ttf',40)
	draw.text((coor[0],coor[1]),name,(255,0,0))#,font=newfont)
	draw.rectangle((coor[0],coor[1],coor[2],coor[3]),outline = "red")
	plt.imshow(img)
	plt.show()

if __name__ == '__main__':
	# for test
	dis_gt(Image.open("/home/pzl/Data/tiny_vid/bird/000001.JPEG"),'bird',np.array([46,0,123,70]))