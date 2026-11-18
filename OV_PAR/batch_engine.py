import time
import numpy as np
import torch
import log_untils
from tools.utils import AverageMeter, to_scalar, time_str

def batch_trainer(epoch, model, train_set, train_loader, optimizer, args, logger):
    model.train()
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    lr = optimizer.param_groups[0]['lr']
    metric_logger = log_untils.MetricLogger(delimiter='  ', logger=logger)
    header = 'Epoch: [{}]'.format(epoch)
    correct, total = (0, 0)
    for step, (imgs, gt_label, imgname) in enumerate(metric_logger.log_every(train_loader, int(batch_num / 2), header)):
        batch_time = time.time()
        imgs, gt_label = (imgs.cuda(non_blocking=True), gt_label.cuda(non_blocking=True))
        logits, cls_loss, contra_loss, L_anchor = model(imgs, imgname, train_set.base_indices, gt_label, epoch)
        if args.stage == 'stage1':
            L_anchor = L_anchor * 500.0
        train_loss = (cls_loss[0] + cls_loss[1] + cls_loss[2]) / 2 + contra_loss
        if args.stage == 'stage1':
            train_loss = train_loss + L_anchor
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss_meter.update(to_scalar(train_loss))
        label = gt_label.cpu().detach().numpy()
        outputs_np = logits.cpu().detach().numpy()
        predicted = outputs_np > 0.5
        correct += np.sum(predicted == label, axis=0)
        total += gt_label.size(0)
        acc = np.mean(correct / total)
        metric_logger.update(loss=train_loss.item())
        if cls_loss[0] != 0:
            metric_logger.update(cls_ca=cls_loss[0].item())
        if cls_loss[1] != 0:
            metric_logger.update(cls_glb=cls_loss[1].item())
        if cls_loss[2] != 0:
            metric_logger.update(cls_loc=cls_loss[2].item())
        if args.stage == 'stage1' and L_anchor != 0:
            metric_logger.update(anchor=L_anchor.item())
        if contra_loss != 0:
            metric_logger.update(cl=contra_loss.item())
        metric_logger.update(acc=acc)
        metric_logger.update(lr=lr)
    return loss_meter.avg

def valid_trainer(epoch, model, val_set, valid_loader, args):
    model.eval()
    model._precompute_text_embeddings()
    preds_probs = []
    novel_preds_probs = []
    gt_list = []
    novel_gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(valid_loader):
            imgs = imgs.cuda(non_blocking=True)
            labels = gt_label.numpy()
            gt_list.append(labels[:, list(val_set.base_indices) + list(val_set.novel_indices)])
            novel_gt_list.append(labels[:, val_set.novel_indices])
            valid_logits, valid_novel_logits, _, _ = model.infer(epoch, imgs, imgname, val_set.base_indices, val_set.novel_indices)
            preds_probs.append(valid_logits.cpu().numpy())
            novel_preds_probs.append(valid_novel_logits.cpu().numpy())
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    novel_gt_label = np.concatenate(novel_gt_list, axis=0)
    novel_preds_probs = np.concatenate(novel_preds_probs, axis=0)
    return (preds_probs, novel_preds_probs, gt_label, novel_gt_label)
