# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import cfg
from ..box_utils import match, log_sum_exp, jaccard, decode


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        if(use_gpu):
            torch.cuda.set_device(cfg['cuda_device'])
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']


    def conf_postive(self, conf_mask, conf_data,pos,num):
        m = nn.Softmax()
        loss_m_ = 0
        for idx in range(num):
            data = conf_data[idx]
            loss_m = data[pos[idx]]
            items = loss_m.shape[0]
            loss_m = m(loss_m)
            loss_m = loss_m*conf_mask[idx]
            loss_m = loss_m.sum(0)
            loss_m = loss_m.sum(0)
            loss_m = loss_m
            loss_m_+=loss_m
        return loss_m_/num
    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        use_loss = False
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        # print(torch.cuda.current_device())
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if(use_loss):
            conf_mask = torch.FloatTensor(num, num_classes)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda(cfg['cuda_device'])
            conf_t = conf_t.cuda(cfg['cuda_device'])

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)

        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        gatherd_conf = batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - gatherd_conf

        # simi_idx = pos.unsqueeze(pos.dim()).expand_as(simi_data)
        # Compute simi feature across batch
        # simi = simi_data[simi_idx].view(-1, 4)
        # simi = simi_data[simi_idx].view(-1, 2)
        # loss_s = 0
        # simi_Loss_s = torch.nn.TripletMarginLoss(margin=1)

        # 重要
        # simi_cos_Loss_s = torch.nn.CosineEmbeddingLoss(margin=0.5)
        #
        # pos_count = num_pos.squeeze(1).sum(0)
        # boxes = []
        # for idx1 in num_pos:
        #     boxes.append(idx1)
        # start_dix = 0
        # k=0
        # one_target = torch.ones(4)
        # one_target_negative = one_target*(-1)
        # prior_simi = defaults.unsqueeze(0).expand_as(loc_data)
        # prior_simi = prior_simi[pos_idx].view(-1, 4);
        # loc_boxes = decode(loc_p,prior_simi,self.variance)
        # for idx in num_pos:
        #     for i in range(start_dix, start_dix+idx-1):
        #         anchor = simi[i].view(-1, 4)
        #         icount = 0
        #         for j in range(start_dix+i, start_dix+idx):
        #             featureID_overlaps_pos = jaccard(
        #                 loc_boxes[i, :].unsqueeze(0),
        #                 loc_boxes[j, :].unsqueeze(0)
        #             )
        #             if featureID_overlaps_pos.squeeze(1).item() < 0.01:
        #                 if icount>0:
        #                     continue
        #                 positive = simi[i].view(-1, 4)
        #                 icount+=1
        #             else:
        #                 positive = simi[j].view(-1, 4)
        #             if start_dix + idx >= pos_count.item():
        #                 neg_num = random.randint(0, idx - 1)
        #                 negative = simi[start_dix - neg_num - 1].view(-1, 4)
        #                 loss_s += simi_cos_Loss_s(anchor, positive, one_target)
        #                 loss_s += simi_cos_Loss_s(anchor, negative, one_target_negative)
        #                 break
        #             else:
        #                 neg_num = boxes[k + 1]
        #                 neg_num = random.randint(0, neg_num - 1)
        #                 negative = simi[start_dix + idx + neg_num].view(-1, 4)
        #                 loss_s += simi_cos_Loss_s(anchor, positive, one_target)
        #                 loss_s += simi_cos_Loss_s(anchor, negative, one_target_negative)
        #     start_dix += idx
        #     k+=1



        # for idx in num_pos:
        #     for i in range(start_dix, start_dix+idx):
        #         anchor = simi[i].view(-1, 4)
        #         # anchor = simi[i].view(-1, 2)
        #         icount = 0
        #         for j in range(start_dix, start_dix+idx):
        #             featureID_overlaps_pos = jaccard(
        #                 loc_boxes[i, :].unsqueeze(0),
        #                 loc_boxes[j, :].unsqueeze(0)
        #             )
        #             # if featureID_overlaps_pos.squeeze(1).item() > 0.5:
        #             #     continue;
        #
        #             # elif featureID_overlaps_pos.squeeze(1).item() == 0:
        #             #     negative = simi[j].view(-1, 4)
        #             #     loss_s += simi_Loss_s(anchor, positive, negative)
        #             #     continue
        #             if featureID_overlaps_pos.squeeze(1).item() < 0.01:
        #                 if icount>0:
        #                     continue
        #                 positive = simi[i].view(-1, 4)
        #                 # positive = simi[i].view(-1, 2)
        #                 icount+=1
        #
        #             else:
        #                 positive = simi[j].view(-1, 4)
        #                 # positive = simi[j].view(-1, 1)
        #             if start_dix + idx >= pos_count.item():
        #                 neg_num = random.randint(0, idx - 1)
        #                 negative = simi[start_dix - neg_num - 1].view(-1, 4)
        #                 # negative = simi[start_dix - neg_num - 1].view(-1, 2)
        #                 loss_s += simi_Loss_s(anchor, positive, negative)
        #                 break
        #             else:
        #                 neg_num = boxes[k + 1]
        #                 neg_num = random.randint(0, neg_num - 1)
        #                 negative = simi[start_dix + idx + neg_num].view(-1, 4)
        #                 loss_s += simi_Loss_s(anchor, positive, negative)
        #     start_dix += idx
        #     k+=1


        # Hard Negative Mining
        # loss_c = loss_c.view(pos.size()[0], pos.size()[1])
        # loss_c[pos] = 0  # filter out pos boxes for now
        loss_c[pos.view(-1,1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # compute mutually exclusive loss
        if(use_loss):
            loss_m = self.conf_postive(conf_mask, conf_data, pos, num)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        # N = num_pos.data.sum().double()
        # loss_l = loss_l.double()
        # loss_c = loss_c.double()
        loss_l /= N.float()
        loss_c /= N.float()
        # loss_s /= N.float()
        if(use_loss):
            return loss_l, loss_c,loss_m
        else:
            return loss_l, loss_c
