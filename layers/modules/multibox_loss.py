# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp


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
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']


    ## 다음에 다시 확인해보기...
    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions

        ## num: batch size.
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes

        ## loc_t: Tensor to be filled w/ endcoded location targets.
        ## conf_t: Tensor to be filled w/ matched indices for conf preds.
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            ## idx: an index of a batch.
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            ## match(threshold, truths, priors, variances, labels,
            ##       loc_t, conf_t, idx)
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
            ## match 함수를 통해 loc_t, conf_t를 구함.

        if self.use_gpu:
            ## loc_t.cuda(): CUDA 메모리 안에 있는 loc_t object의 copy를 리턴..
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        ## requires_grad=False: 이 변수에 대해 gradient를 계산하지 않는다.
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        ## match된 proir의 박스들..
        ## conf_t[k] == 0: background labels..
        pos = conf_t > 0
        ## pos (Shape): [batch, num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)
        # Localization Loss (Smooth L1)
        ## pos.unsqueeze(pos.dim()): (Shape) [batch, num_priors, 1]
        ## pos.unsqueeze(pos.dim()).expand_as(loc_data): (Shape)
        ##                                               [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        ## expand_as를 하여 [batch, num_priors, 1]을 [batch, num_priors, 4]로
        ## 늘린다. 이유는 박스를 정의하는 4개의 값을 사용하기 위해서..

        ## Flatten [batch_size, num_priors, 4] to [batch_size * num_priors, 4]
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        ## Localization loss 계산.
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        ## conf_data : (Shape) [batch_size, num_priors, num_classes]
        ## batch_conf : (Shape) [batch_size * num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        ## log_sum_exp(batch_conf) : (Shape) [batch_size * num_priors, 1]
        ## batch_conf.gather(1, conf_t.view(-1, 1)) : (Shape) [batch_size * num_priors, 1]
        ## loss_c : (Shape) [batch_size * num_priors, 1]

        ## 다음에 확인해 보기..
        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
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

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
