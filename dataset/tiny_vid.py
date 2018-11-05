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

def train_test_txt(defualt_path='/home/pzl/Data/tiny_vid'):
    """
    将如下格式存入文件：
        /home/pzl/Data/tiny_vid/turtle/000151.JPEG  2   29 38 108 84
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

class tiny_vid_loader(data.Dataset):
    """
    功能：
        构造一个用于tiny_vid数据集的迭代器
    参数：

    """
    def __init__(self,defualt_path='/home/pzl/Data/tiny_vid',mode='train',transform=None):#SSDAugmentation(size=128)):
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
                self.class_coor.append([int(i) for i in line[2:]])
        self.transform_default = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform = transform


    def __getitem__(self, index):
        imgpath = self.filelist[index]
        gt_class = np.array(self.class_coor[index][-1],dtype = np.int64)
        gt_bbox = np.array(self.class_coor[index][:-1],dtype = np.float32)
        if  self.transform is None:
            img = Image.open(imgpath).convert('RGB')
            img = self.transform_default(img)
        else:
            # bgr
            img = (cv2.imread(imgpath)).astype(np.float32)
            gt_bbox = gt_bbox[np.newaxis,:]
            # data augmentation
            img,gt_bbox = self.transform(img,gt_bbox)
            # print('img.shape:',img.shape)
            # print('gt_bbox.shape',gt_bbox.shape)
            # to rgb to Image
            img = img[:, :, (2, 1, 0)]
            img = Image.fromarray(np.uint8(img))
            img = self.transform_default(img)
            # print('img.size',img.size())

            # to normalize
        return img,torch.tensor(gt_class.astype(np.int64)),torch.tensor(gt_bbox.astype(np.float64))

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
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w,h = img.size #(128,128)
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target,boxes,labels

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

if __name__ == '__main__':
    target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
    dst = tiny_vid_loader()#transform = None)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, (img, gt_class,gt_bbox) in enumerate(trainloader):
        if i % 150 == 0:
            # print(gt_class)
            # print(gt_bbox)
            # print(img)
            gt_class=gt_class.data.numpy()
            gt_bbox = gt_bbox.data.numpy()[0]
            inp = img.numpy()[0].transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)*255
            inp = Image.fromarray(inp.astype('uint8')).convert('RGB')
            dis_gt(inp,target_classes[gt_class],gt_bbox)

            # break
    