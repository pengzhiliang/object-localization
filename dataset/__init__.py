#-*-coding:utf-8-*-
from tiny_vid import *

def xywh_to_x1y1x2y2(boxes):
    # xywh转为四个框的坐标
    new_boxes = boxes.clone()

    new_boxes[:,0] = boxes[:,0]-boxes[:,2]
    new_boxes[:,1] = boxes[:,1]-boxes[:,3]
    new_boxes[:,2] = boxes[:,2]+boxes[:,0]
    new_boxes[:,3] = boxes[:,3]+boxes[:,1]

    return new_boxes

def x1y1x2y2_to_xywh(boxes):
    # 与上面函数相反
    new_boxes = boxes.clone()
    new_boxes[:,0] = (boxes[:,0]+boxes[:,2])/2.
    new_boxes[:,1] = (boxes[:,1]+boxes[:,3])/2.
    new_boxes[:,2] = (boxes[:,2]-boxes[:,0])/2.
    new_boxes[:,3] = (boxes[:,3]-boxes[:,1])/2.

    return new_boxes