import os
from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict

from tools.utils import may_mkdirs
import os
from datetime import datetime
import torch.distributed as dist
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:.2f}M")


def create_exp_dir(args, path):
    if args.distributed:
        # 确保只有一个进程创建文件夹
        if dist.get_rank() == 0:
            if args.exp:
                exp_dir = os.path.join(path, args.exp)
                index = 1
                while os.path.exists(exp_dir):
                    exp_dir = os.path.join(path, f"{args.exp}_{index}")
                    index += 1
                os.makedirs(exp_dir)
            else:
                now = datetime.now()
                timestamp = now.strftime("%m_%d_%H_%M")
                exp_dir = os.path.join(path, timestamp)
                os.makedirs(exp_dir)
    else:
        if args.exp:
            exp_dir = os.path.join(path, args.exp)
            index = 1
            while os.path.exists(exp_dir):
                exp_dir = os.path.join(path, f"{args.exp}_{index}")
                index += 1
            os.makedirs(exp_dir)
        else:
            now = datetime.now()
            timestamp = now.strftime("%m_%d_%H_%M")
            exp_dir = os.path.join(path, timestamp)
            os.makedirs(exp_dir)
    
    return exp_dir


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

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


def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.45):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result


import numpy as np
from easydict import EasyDict

def get_pedestrian_metrics_single(gt_label, preds_probs, threshold=0.45):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    num_classes = gt_label.shape[1]
    
    label_acc = np.zeros(num_classes)
    label_f1 = np.zeros(num_classes)

    for c in range(num_classes):
        # TP + FN
        gt_pos = np.sum(gt_label[:, c] == 1).astype(float)
        # TN + FP
        gt_neg = np.sum(gt_label[:, c] == 0).astype(float)
        # TP
        true_pos = np.sum((gt_label[:, c] == 1) & (pred_label[:, c] == 1)).astype(float)
        # TN
        true_neg = np.sum((gt_label[:, c] == 0) & (pred_label[:, c] == 0)).astype(float)
        # FP
        false_pos = np.sum((gt_label[:, c] == 0) & (pred_label[:, c] == 1)).astype(float)
        # FN
        false_neg = np.sum((gt_label[:, c] == 1) & (pred_label[:, c] == 0)).astype(float)

        label_acc[c] = (true_pos + true_neg) / (gt_pos + gt_neg + eps)
        
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        label_f1[c] = 2 * precision * recall / (precision + recall + eps)
    result.label_acc = label_acc
    result.label_f1 = label_f1

    return result
