import os
from collections import OrderedDict

import numpy as np
import torch

from tools.utils import may_mkdirs


def seperate_weight_decay(named_params, lr, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        # if 'bias' in name:
        #     no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'lr': lr, 'weight_decay': 0.},
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)

    # --------------------- dangwei li TIP20 ---------------------
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)


    # --------------------- AAAI ---------------------
    # pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
    # neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
    # weights = pos_weights + neg_weights

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights


def get_model_log_path(root_path, model_name):
    multi_attr_model_dir = os.path.join(root_path, model_name, 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, model_name, 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir


class LogVisual:

    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.val_loss = []

        self.ap = []
        self.map = []
        self.acc = []
        self.prec = []
        self.recall = []
        self.f1 = []

        self.error_num = []
        self.fn_num = []
        self.fp_num = []

        self.save = False

    def append(self, **kwargs):
        self.save = False

        if 'result' in kwargs:
            self.ap.append(kwargs['result']['label_acc'])
            self.map.append(np.mean(kwargs['result']['label_acc']))
            self.acc.append(np.mean(kwargs['result']['instance_acc']))
            self.prec.append(np.mean(kwargs['result']['instance_precision']))
            self.recall.append(np.mean(kwargs['result']['instance_recall']))
            self.f1.append(np.mean(kwargs['result']['floatance_F1']))

            self.error_num.append(kwargs['result']['error_num'])
            self.fn_num.append(kwargs['result']['fn_num'])
            self.fp_num.append(kwargs['result']['fp_num'])

        if 'train_loss' in kwargs:
            self.train_loss.append(kwargs['train_loss'])
        if 'val_loss' in kwargs:
            self.val_loss.append(kwargs['val_loss'])


def get_pkl_rootpath(dataset, zero_shot):
    root = os.path.join("/data/jinjiandong/dataset/MSP_degrade", f"{dataset}")
    if zero_shot:
        data_path = os.path.join(root, 'dataset_zs_run0.pkl')
    else:
        data_path = os.path.join(root, 'dataset_random.pkl')  #

    return data_path


def get_reload_weight(model_path, model, pth='ckpt_max.pth'):
    model_path = os.path.join(model_path, pth)
    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")

    model.load_state_dict(pretrain_dict, strict=True)

    return model
