import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("scaledbceloss")
class ScaledBCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=30, tb_writer=None):
        super(ScaledBCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.pos_scale = scale
        self.neg_scale = scale
        self.tb_writer = tb_writer

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        logits = logits * targets * self.pos_scale + logits * (1 - targets) * self.neg_scale

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))

        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()

        return [loss], [loss_m]