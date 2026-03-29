""" Cosine Scheduler
Cosine LR schedule with warmup, cycle/restarts, noise.
Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

from timm.scheduler.scheduler import Scheduler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.
    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        """

        @param optimizer:
        @param t_initial:  epoch number of the first cosine decay iteration
        @param t_mul:  multiplier between multiple cosine decay iterations
        @param lr_min: final learning rate
        @param decay_rate: decay rate between the peak values of multiple cosine decay iterations
        @param warmup_t: the epoch number of warmup stage
        @param warmup_lr_init: the initial learning rate of warmup stage
        @param warmup_prefix:
        @param cycle_limit: the iteration limit number of
        @param t_in_epochs:
        @param noise_range_t:
        @param noise_pct:
        @param noise_std:
        @param noise_seed:
        @param initialize:
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                            "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                # math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul) < 1 always hold
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                # 0.5 * (1 + math.cos(math.pi * t_curr / t_i)), the proportion; (lr_max - lr_min), lr scale
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in
                    lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))


if __name__ == '__main__':

    params = torch.zeros(10)
    lr = 2e-4

    optim = torch.optim.SGD([params], lr, momentum=0.9, weight_decay=5e-4)

    # num_epochs = 160

    # scheduler = CosineLRScheduler(
    #     optim,
    #     t_initial=30,
    #     t_mul=1,  # cosine decay epoch multiplier
    #     # lr_min=1e-5,  # cosine lr 最终回落的位置
    #     decay_rate=0.5,
    #     # warmup_lr_init=1e-5,
    #     # warmup_t=3,
    #     cycle_limit=3,  # 最大的限制
    #     # t_in_epochs=True,
    #     # noise_range_t=None,
    #     # noise_pct=0.67,
    #     # noise_std=1,
    #     # noise_seed=42
    # )

    # be called .step() after every batch
    scheduler = lr_scheduler.OneCycleLR(optim, max_lr=lr, steps_per_epoch=641, epochs=40,
                                        pct_start=0.0)

    # plt.figure(figsize=(8, 8))
    lr = []
    lr_s = []
    for i in range(40):
        for j in range(641):
            lr.append(optim.param_groups[0]['lr'])
            lr_s.append(scheduler.get_last_lr()[0])
            optim.step()
            scheduler.step()
    plt.plot(range(40 * 641), lr)
    plt.show()

    plt.plot(range(40 * 641), lr_s)
    plt.show()
