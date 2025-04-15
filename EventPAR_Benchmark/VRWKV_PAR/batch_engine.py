import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import log_untils
from tools.utils import AverageMeter, to_scalar, time_str
def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()

    epoch_time = time.time()
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    lr = optimizer.param_groups[0]['lr']
    metric_logger = log_untils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    for step, (img_name,imgs, gt_label, label_v) in enumerate(metric_logger.log_every(train_loader, int(batch_num/3), header)):  
       
        batch_time = time.time()
        imgs = [i.cuda() for i in imgs]
        gt_label = gt_label.cuda()
        label_v = label_v[0].cuda()

        train_logits = model(imgs, label_v)
        train_loss = criterion(train_logits, gt_label)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) 
        optimizer.step()
        loss_meter.update(to_scalar(train_loss))
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())
        metric_logger.update(train_loss=train_loss.item())
        metric_logger.update(VTB_lr=lr)

   
    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return train_loss, gt_label, preds_probs


def valid_trainer(model, valid_loader, criterion):
   
    model.eval()
    loss_meter = AverageMeter()
    metric_logger = log_untils.MetricLogger(delimiter="  ")
    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (img_name,imgs, gt_label, label_v) in enumerate(valid_loader):
            imgs = [i.cuda() for i in imgs]
            label_v = label_v[0].cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits = model(imgs, label_v)
            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))
            metric_logger.update(valid_loss=valid_loss.item())


    valid_loss = loss_meter.avg
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs