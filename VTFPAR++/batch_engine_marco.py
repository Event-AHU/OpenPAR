import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.nn as nn
from tools.utils import AverageMeter, to_scalar, time_str

img_count=0

# === 辅助：按旧编码边界切分，再对齐到新编码 ===
def split_gt_to_indices(gt_onehot: torch.Tensor, attr_len, dataset: str):
    B, D_old = gt_onehot.shape
    if dataset == 'MARS':
        old_boundaries = [0,1,2,3,4,5,6,7,8,9,15,20,29,39,43]
    else:
        old_boundaries = [0,1,2,3,4,5,6,7,8,14,19,28,36]

    old_lens = [old_boundaries[i+1] - old_boundaries[i] for i in range(len(old_boundaries)-1)]

    old_slices = torch.split(gt_onehot, old_lens, dim=1)
    aligned = []
    for j, (src, al) in enumerate(zip(old_slices, attr_len)):
        w_src = src.size(1)
        if w_src == al:
            aligned.append(src)
        elif w_src == 1 and al == 2:
            p = src
            aligned.append(torch.cat([1 - p, p], dim=1))

    targets = [torch.argmax(t, dim=1) for t in aligned]
    return targets  # List[Tensor(B,)]

def batch_trainer(epoch, model, ViT_model, train_loader, criterion, optimizer, args):
    model.train()
    ViT_model.train()
    loss_meter = AverageMeter()
    if args.dataset == 'MARS':
        attr_len = [2,2,2,2,2,2,2,2,2,6,5,9,10,4]
        attr_act_len = [1,1,1,1,1,1,1,1,1,6,5,9,10,4]
        group = "top length, bottom type, shoulder bag, backpack, hat, hand bag, hair, gender, bottom length, pose, motion, top color, bottom color, age"
    else:
        attr_len = [2,2,2,2,2,2,2,6,5,9,8]
        attr_act_len = [1,1,1,1,1,1,1,6,5,9,8]
        group = "backpack, shoulder bag, hand bag, boots, gender, hat, shoes, top length, pose, motion, top color, bottom color"
    criterion = nn.CrossEntropyLoss()
    preds_list = [[] for _ in range(len(attr_len))]
    gt_list = [[] for _ in range(len(attr_len))]
    # 根据 index_list 计算 attr_lens
    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        imgs = imgs.cuda()
        gt_label = gt_label.cuda()   # 保证每个属性是一维单标签 

        outputs = model(imgs, ViT_model=ViT_model)  # 多分支输出
        attrs = split_gt_to_indices(gt_label, attr_len, args.dataset)  # 转为 index (B,)
        attrs = torch.stack(attrs).cuda() 
        # === 2. 按 attr_len 切分 outputs ===
        split_outputs = torch.split(outputs, attr_len, dim=1)  # list，每个是 (B, attr_len[i])

        # === 3. 逐属性计算 CE loss ===
        loss = 0
        for i, (out, target, al) in enumerate(zip(split_outputs, attrs, attr_len)):
            loss += criterion(out, target) / al

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        # 保存预测结果（取 argmax）
        for i in range(len(split_outputs)):
            pred = torch.argmax(split_outputs[i], 1).cpu().numpy()
            label = attrs[i].cpu().numpy()
            preds_list[i].extend(pred)
            gt_list[i].extend(label)

    print(f"Epoch {epoch}, Train Loss {loss_meter.avg:.4f}")
    return loss_meter.avg, gt_list, preds_list


from sklearn.metrics import f1_score, accuracy_score

def valid_trainer(epoch, model, ViT_model, valid_loader, criterion, args):
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    if args.dataset == 'MARS':
        attr_len = [2,2,2,2,2,2,2,2,2,6,5,9,10,4]
        group = "top length, bottom type, shoulder bag, backpack, hat, hand bag, hair, gender, bottom length, pose, motion, top color, bottom color, age"
    else:
        attr_len = [2,2,2,2,2,2,2,6,5,9,8]
        group = "backpack, shoulder bag, hand bag, boots, gender, hat, shoes, top length, pose, motion, top color, bottom color"

    criterion = nn.CrossEntropyLoss()

    preds_list = [[] for _ in range(len(attr_len))]
    gt_list = [[] for _ in range(len(attr_len))]
    accs = np.array([0 for _ in range(len(attr_len))])
    num = 0
    with torch.no_grad():
        for imgs, gt_label, _ in valid_loader:
            num += imgs.shape[0]
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()   # (B, 41) one-hot

            # === 1. 标签切分并转 index ===
            attrs = split_gt_to_indices(gt_label, attr_len, args.dataset)  # 转为 index (B,)
            attrs = torch.stack(attrs).cuda() # (B, 41)

            # === 2. 输出切分 ===
            outputs = model(imgs, ViT_model=ViT_model)   # (B, 48)
            split_outputs = torch.split(outputs, attr_len, dim=1)

            # === 3. 计算 loss ===
            loss = 0
            for i, (out, target, al) in enumerate(zip(split_outputs, attrs, attr_len)):
                loss += criterion(out, target) / al
            loss_meter.update(loss.item())

            # === 4. 保存预测和标签 ===
            for i in range(len(split_outputs)):
                pred = torch.argmax(split_outputs[i], 1).cpu().numpy()
                label = attrs[i].cpu().numpy()
                preds_list[i].extend(pred)
                gt_list[i].extend(label)
                accs[i] += np.sum(label == pred)
    # === 5. 逐属性 acc/f1 ===
    # accs = [accuracy_score(gt_list[i], preds_list[i]) for i in range(len(preds_list))]
    accs = accs / num
    # avr = np.mean(accs)
    f1_macros = [f1_score(gt_list[i], preds_list[i], average="macro") for i in range(len(preds_list))]
    f1_micros = [f1_score(gt_list[i], preds_list[i], average="micro") for i in range(len(preds_list))]

    print("Validation Results")
    print("Accs:", accs)
    print("Macro-F1:", f1_macros)
    print("Micro-F1:", f1_micros)
    print(f"Avg Acc {np.mean(accs):.4f}, Avg Macro-F1 {np.mean(f1_macros):.4f}, Avg Micro-F1 {np.mean(f1_micros):.4f}")

    return loss_meter.avg, accs, f1_macros, f1_micros
