from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight


class CEL_Sigmoid(nn.Module):
    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(CEL_Sigmoid, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self, logits, targets, k=None):
        batch_size = logits.shape[0]

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')    

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx], self.sample_weight)
                loss = loss[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            # import pdb;pdb.set_trace()
            loss = (loss * weight.cuda())
        if k is not None:
            sample_loss = loss.sum(dim=1)
            topk_values, topk_indices = torch.topk(sample_loss, k=k, dim=0, largest=True)

        loss = loss.sum() / batch_size if self.size_average else loss.sum()
        if k is not None:
            return loss, topk_indices
        else:
            return loss
