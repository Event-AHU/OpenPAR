import os
from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict

from tools.utils import may_mkdirs


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)
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


def get_pkl_rootpath(dataset):
    root = os.path.join("/dataset", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path


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

def get_signle_metrics(gt_label, preds_probs, threshold=0.45):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1)).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0)).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1)).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0)).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1))).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0))).astype(float)

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

    result.instance_f1 = np.mean(result.label_f1)
    result.instance_acc = np.mean(label_ma)
    result.instance_prec = np.mean(result.label_prec)
    result.instance_recall = np.mean(result.label_pos_recall)

    #result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

def simple_par_metrics(gt_label, preds_probs, threshold=0.45, signle=False):
    pred_label = preds_probs > threshold
        
    eps = 1e-20
    result = EasyDict()
    if signle:
        preds_probs = preds_probs[:,np.newaxis]
    # TP
    TP = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float) 
    # TN
    TN = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    FP = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    FN = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    TP =  np.sum(TP).astype(float)
    TN =  np.sum(TN).astype(float)
    FP =  np.sum(FP).astype(float)
    FN =  np.sum(FN).astype(float)

    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    prec = (TP) / (TP + FP + eps)
    recall = (TP) / (TP + FN + eps)
    f1 = 2 * prec * recall / (prec + recall + eps)
    
    result.acc = acc
    result.prec = prec
    result.recall = recall
    result.f1 = f1

    return result


