#-*-coding:utf-8-*-
'''
Created on Oct 29,2018

@author: pengzhiliang
'''
from __future__ import print_function
import torch,os,sys,random,cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from matplotlib import pyplot as plt
from encoder import DataEncoder
from torch.utils import data
from torchvision import transforms
from os.path import join as pjoin
# from augmentation import SSDAugmentation

#=============================================================================
#
#           Create train_images.txt and test_images.txt
#
#=============================================================================
def train_test_txt(defualt_path='/home/pzl/Data/tiny_vid'):
    """
    将如下格式存入文件：
        /home/pzl/Data/tiny_vid/turtle/000151.JPEG  1  29 38 108 84 2
    其中参数：
        /home/pzl/Data/tiny_vid/turtle/000151.JPEG：图片路径
        1 ： 一个物体（方便使用ssd架构）
        29 38 108 84：分别为 xmin, ymin, xmax, ymax，表示bounding box的位置
        2 ： 种类
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
                f.write('\t'.join([imgpath,'1',imgbbox,imageclass])+'\n')
        with open(pjoin(defualt_path,'test_images.txt'),'a') as f:
            for i in range(151,181): 
                imgname = '000000'
                pad0 = 6 - len(str(i))
                imgname = imgname[:pad0]+str(i)+'.JPEG'
                imgpath = pjoin(pjoin(defualt_path,dirname),imgname)
                imageclass = str(classes[dirname])
                imgbbox = ' '.join(bbox_dic[str(i)])
                f.write('\t'.join([imgpath,'1',imgbbox,imageclass])+'\n')

#=============================================================================
#
#           display a image with boundingbox and class
#
#=============================================================================
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

#=============================================================================
#
#           Create a data loader for two-output net(not ssd),don't need to encode
#
#=============================================================================
class tiny_vid_loader(data.Dataset):
    """
    功能：
        构造一个用于tiny_vid数据集的迭代器
    参数：

    """
    img_size =128

    def __init__(self,defualt_path='/home/pzl/Data/tiny_vid',mode='train',transform='some augmentation'):
        """
        defualt_path: 如'/home/pzl/Data/tiny_vid'
        mode : 'train' or 'test'
        """
        if not (os.path.exists(pjoin(defualt_path,'train_images.txt')) and os.path.exists(pjoin(defualt_path,'test_images.txt'))):
            train_test_txt(defualt_path)
        self.filelist=[]
        self.class_coor = []
        self.mode = True if mode =='train' else False
        # /home/pzl/Data/tiny_vid/turtle/000151.JPEG  1  29 38 108 84 2
        with open(pjoin(defualt_path,mode+'_images.txt')) as f:
            for line in f.readlines():
                line = line.strip().split()
                self.filelist.append(line[0])
                self.class_coor.append([int(i) for i in line[2:]])
        self.ToTensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.data_encoder = DataEncoder()
        self.transform = transform

    def random_distort( self,
        img,
        brightness_delta=32/255.,
        contrast_delta=0.5,
        saturation_delta=0.5,
        hue_delta=0.1):
        '''A color related data augmentation used in SSD.

        Args:
          img: (PIL.Image) image to be color augmented.
          brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
          contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
          saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
          hue_delta: (float) shift of hue, range from [-delta,delta].

        Returns:
          img: (PIL.Image) color augmented image.
        '''
        def brightness(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(brightness=delta)(img)
            return img

        def contrast(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(contrast=delta)(img)
            return img

        def saturation(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(saturation=delta)(img)
            return img

        def hue(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(hue=delta)(img)
            return img

        img = brightness(img, brightness_delta)
        if random.random() < 0.5:
            img = contrast(img, contrast_delta)
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
        else:
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
            img = contrast(img, contrast_delta)
        return img

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.
        只在水平方向翻转
        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[2]
            xmax = w - boxes[0]
            boxes[0] = xmin
            boxes[2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.

        For more details, see 'Chapter2.2: Data augmentation' of the paper.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [4,].
          labels: (tensor) bbox labels, sized [1,].

        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        boxes = torch.unsqueeze(boxes,dim=0) # expand [1,4]
        # print(boxes)
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                # random.randrange(min,max)包含min 不包含max
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x+w, y+h))
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                # print(selected_boxes, mask)
                return img, selected_boxes, labels#labels[mask]

    def __getitem__(self, index):
        imgpath = self.filelist[index]
        gt_class = np.array(self.class_coor[index][-1],dtype = np.float32)
        gt_bbox = np.array(self.class_coor[index][:-1],dtype = np.float32)
        # print('1:',gt_bbox)
        img = Image.open(imgpath).convert('RGB')
        if  self.transform is not None:
            gt_class , gt_bbox = torch.Tensor(gt_class),torch.Tensor(gt_bbox)
            # print('2:',gt_class)
            if self.mode:
                img = self.random_distort(img)
                img , gt_bbox = self.random_flip(img,gt_bbox)
                img, gt_bbox, gt_class = self.random_crop(img, gt_bbox, gt_class)
                w,h = img.size 
                gt_bbox /= torch.Tensor([w,h,w,h]).expand_as(gt_bbox)
                # print('3:',gt_bbox*128)
                img = transforms.Resize((128,128))(img)
            img = self.ToTensor(img)
            img = self.Normalize(img)
        else:
            img,gt_class , gt_bbox = self.ToTensor(img),torch.Tensor(gt_class),torch.Tensor(gt_bbox/128.)
            img = self.Normalize(img)
        return img,gt_class.long(),(gt_bbox*128).squeeze()

    def __len__(self):
        return len(self.filelist)


class ListDataset(data.Dataset):
    img_size = 300

    def __init__(self, root, list_file, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.train = train
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            num_objs = int(splited[1])
            box = []
            label = []
            for i in range(num_objs):
                xmin = splited[2+5*i]
                ymin = splited[3+5*i]
                xmax = splited[4+5*i]
                ymax = splited[5+5*i]
                c = splited[6+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load a image, and encode its bbox locations and class labels.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = Image.open(fname)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Data augmentation while training.
        if self.train:
            img =self.random_distort(img)
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w,h = img.size 
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target,boxes,labels

    def random_distort(self,
        img,
        brightness_delta=32/255.,
        contrast_delta=0.5,
        saturation_delta=0.5,
        hue_delta=0.1):
        '''A color related data augmentation used in SSD.

        Args:
          img: (PIL.Image) image to be color augmented.
          brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
          contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
          saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
          hue_delta: (float) shift of hue, range from [-delta,delta].

        Returns:
          img: (PIL.Image) color augmented image.
        '''
        def brightness(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(brightness=delta)(img)
            return img

        def contrast(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(contrast=delta)(img)
            return img

        def saturation(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(saturation=delta)(img)
            return img

        def hue(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(hue=delta)(img)
            return img

        img = brightness(img, brightness_delta)
        if random.random() < 0.5:
            img = contrast(img, contrast_delta)
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
        else:
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
            img = contrast(img, contrast_delta)
        return img

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.

        For more details, see 'Chapter2.2: Data augmentation' of the paper.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].

        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                # random.randrange(min,max)包含min 不包含max
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x+w, y+h))
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                return img, selected_boxes, labels[mask]

    def __len__(self):
        return self.num_samples

