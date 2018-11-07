from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLoss(nn.Module):
    num_classes = 21


    def __init__(self,device_ids=[0,1,2,3]):
        super(MultiBoxLoss, self).__init__()
        self.device_ids = device_ids
        self.loc_loss_layer = nn.SmoothL1Loss(reduction='sum')
        self.conf_loss_layer = nn.CrossEntropyLoss(reduction='sum')
        if len(device_ids) != 1:
            self.loc_loss_layer = nn.DataParallel(self.loc_loss_layer)
            self.loc_loss_layer.cuda()
            self.conf_loss_layer = nn.DataParallel(self.conf_loss_layer)
            self.conf_loss_layer.cuda()
    def cross_entropy_loss(self, x, y):
        '''Cross entropy loss w/o averaging across all samples.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) cross entroy loss, sized [N,].
        '''
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1,1))

    def test_cross_entropy_loss(self):
        a = Variable(torch.randn(10,4))
        b = Variable(torch.ones(10).long())
        loss = self.cross_entropy_loss(a,b)
        print(loss.mean())
        print(F.cross_entropy(a,b))

    def hard_negative_mining(self, conf_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N,8732].

        Return:
          (tensor) negative indices, sized [N,8732].
        '''
        batch_size, num_boxes = pos.size()

        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]

        _,idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _,rank = idx.sort(1)  # [N,8732]

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_boxes-1)  # [N,1]

        neg = rank < num_neg.expand_as(rank)  # [N,8732]
        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        '''Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, 8732, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, 8732, 4].
          conf_preds: (tensor) predicted class confidences, sized [batch_size, 8732, num_classes].
          conf_targets: (tensor) encoded target classes, sized [batch_size, 8732].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets).
        '''
        batch_size, num_boxes, _ = loc_preds.size()

        pos = conf_targets>0  # [N,8732], pos means the box matched.
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]))

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)    # [N,8732,4]
        pos_loc_preds = loc_preds[pos_mask].view(-1,4)      # [#pos,4]
        pos_loc_targets = loc_targets[pos_mask].view(-1,4)  # [#pos,4]
        loc_loss = self.loc_loss_layer(pos_loc_preds, pos_loc_targets)

        ################################################################
        # conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
        #           + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)
        ################################################################
        conf_loss = self.cross_entropy_loss(conf_preds.view(-1,self.num_classes), \
                                            conf_targets.view(-1))  # [N*8732,]
        neg = self.hard_negative_mining(conf_loss, pos)    # [N,8732]

        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        mask = (pos_mask+neg_mask).gt(0)

        pos_and_neg = (pos+neg).gt(0)
        preds = conf_preds[mask].view(-1,self.num_classes)  # [#pos+#neg,21]
        targets = conf_targets[pos_and_neg]                 # [#pos+#neg,]
        conf_loss = self.conf_loss_layer(preds, targets)

        loc_loss /= num_matched_boxes
        conf_loss /= num_matched_boxes
        print('%f %f' % (loc_loss.data[0], conf_loss.data[0]), end=' ')
        return loc_loss + conf_loss
        
class SSDLoss(nn.Module):
    def __init__(self, num_classes):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].
        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]

        num_neg = 3*pos.sum(1)  # [N,]
        neg = rank < num_neg[:,None]   # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction='sum')

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), \
                                   cls_targets.view(-1), reduction='none')  # [N*#anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets<0] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos|neg].sum()

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos))#, end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss