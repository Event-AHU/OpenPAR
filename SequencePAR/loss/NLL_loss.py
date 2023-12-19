from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight

class NLL_class_weight_loss(nn.Module):
    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(NLL_class_weight_loss, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self,logits, targets,gt_label):

        # 根据真实标签选择对应的对数概率
        selected_log_probs = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        targets_mask = torch.where(gt_label.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
       
        if self.sample_weight is not None:
            weights = ratio2weight(targets_mask, self.sample_weight)       
        # 加权计算损失
        weighted_loss = -selected_log_probs * weights.contiguous().view(-1).cuda()
        
        # 计算平均损失
        loss = torch.mean(weighted_loss)
        
        return loss