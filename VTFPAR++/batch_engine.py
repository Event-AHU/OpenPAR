import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str

img_count=0

def batch_trainer(epoch, model, ViT_model, train_loader, criterion, optimizer):
    global img_count
    model.train()
    ViT_model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    save_name=[]
    lr = optimizer.param_groups[0]['lr']
    print(f'learning rate whith VTB:{lr}')

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        for elem in imgname :
            save_name.append(elem)
        img_count+=imgs.shape[0]
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()

        train_logits = model(imgs,ViT_model=ViT_model)
        train_loss = criterion(train_logits, gt_label)
        optimizer.zero_grad()

        train_loss.backward()
        optimizer.step()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 500
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')
    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f},img_num:{img_count}')
    img_count=0
    return train_loss, gt_label, preds_probs


def valid_trainer(epoch,model,ViT_model, valid_loader, criterion):
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()
    batch_num = len(valid_loader)
    preds_probs = []
    gt_list = []
    save_name=[]
    save_dir=[]
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(valid_loader):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            valid_logits = model(imgs,ViT_model=ViT_model )
            
            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))
    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
