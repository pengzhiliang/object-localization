import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from transforms import resize
from datasets import ListDataset
from evaluations.voc_eval import voc_eval
from models.ssd import SSD300, SSDBoxCoder

from PIL import Image


print('Loading model..')
net = SSD300(num_classes=6)
net.load_state_dict(torch.load('checkpoint/best_ssd.pth'))
net.cuda()
net.eval()

print('Preparing dataset..')
img_size = 300
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels

dataset = istDataset(list_file="/home/pzl/Data/tiny_vid/test_images.txt",
                      transform=transform_test)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
box_coder = SSDBoxCoder()

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

def eval(net, dataset):
    with torch.no_grad():
        for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
            print('%d/%d' % (i, len(dataloader)))
            gt_boxes.append(box_targets.squeeze(0))
            gt_labels.append(label_targets.squeeze(0))

            loc_preds, cls_preds = net(inputs.cuda())
            box_preds, label_preds, score_preds = box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=0.01)

            pred_boxes.append(box_preds)
            pred_labels.append(label_preds)
            pred_scores.append(score_preds)

        print voc_eval(
            pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels, gt_difficults=None,
            iou_thresh=0.5, use_07_metric=True)

eval(net, dataset)
