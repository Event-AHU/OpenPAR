import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str


def batch_trainer(epoch, model, train_loader, criterion, optimizer, args):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']
    if args.accumulate:
        optimizer.zero_grad()
        train_loss_ = 0



    for step, (imgs, gt_label, label_v,_) in enumerate(train_loader):
        batch_time = time.time()

        label_v = label_v[0].cuda()
        if isinstance(imgs, list):
            rgbs, events = imgs[0], imgs[1]
            rgbs, events, gt_label = rgbs.cuda(), events.cuda(), gt_label.cuda()
            train_logits = model(rgbs, events, label_v)
        else: 
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            train_logits = model(imgs, None, label_v)
        
        train_loss = criterion(train_logits, gt_label)
        if args.accumulate:
            train_loss = train_loss / args.accumulation_steps
            train_loss.backward()
            train_loss_ += train_loss
            # 每累积 accumulation_steps 次，更新一次权重
            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 清零梯度
                loss_meter.update(to_scalar(train_loss_))
                train_loss_ = 0
        else:
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_meter.update(to_scalar(train_loss))
        
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 2000
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/8:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')
 

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs

def mix_batch_trainer(epoch, model, train_loader, criterions, optimizer, args):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = [[] for _ in range(len(args.dataset))]
    preds_probs = [[] for _ in range(len(args.dataset))]

    lr = optimizer.param_groups[0]['lr']
    # if args.accumulate:
    #     optimizer.zero_grad()
    #     train_loss_ = 0
    cache = {'imgs': [[] for _ in range(len(args.dataset))],
         'gt_labels': [[] for _ in range(len(args.dataset))],
         'label_v':  [[] for _ in range(len(args.dataset))],
         'dataset_ids': [[] for _ in range(len(args.dataset))]}
    lossrate = []
    for i in range(len(args.dataset)):
        cache['label_v'][i].append(None)
        lossrate.append(1)
    if args.islossrate:
        lossrate = args.lossrate
    N_list, attr_num = train_loader.dataset.get_N()
    for step, (imgs, gt_labels, label_vs, dataset_ids, mask_id, _) in enumerate(train_loader):
        imgs,  gt_labels, label_vs = imgs.cuda(),  gt_labels.cuda(), label_vs.cuda()
        
        for i in range(len(args.dataset)):
            mask1 = mask_id == i
            mask2 = dataset_ids == i
            if sum(mask2) == 0:
                continue
            imgs_i = imgs[mask1]
            gt_labels_i, label_vs_i = gt_labels[mask2], (label_vs[mask2])[0]
            cache['imgs'][i].extend(imgs_i)
            cache['gt_labels'][i].extend(gt_labels_i)
            cache['label_v'][i][0] = label_vs_i
            max_p = N_list[i] * args.batchsize
            N = N_list[i]
            batch_time = time.time()
            while len(cache['imgs'][i]) >= max_p:
                batch_imgs = torch.stack(cache['imgs'][i][:max_p])
                batch_gt_label = torch.stack(cache['gt_labels'][i][:args.batchsize])[:, :attr_num[i]]

                # label_vec = (cache['label_v'][i][0])[:attr_num[i]]
                if args.visual_only:
                    # 创建占位符word_vec，只保留形状信息
                    # label_vs_i的形状是 [num_attributes, feature_dim]
                    if label_vs_i is not None:
                        # 使用零张量作为占位符，保持相同形状
                        label_vec = torch.zeros_like(label_vs_i[:attr_num[i]])
                    else:
                        # 根据数据集创建适当形状的占位符
                        if args.dataset[i] == 'PA100k':
                            label_vec = torch.zeros(26, 768)
                        elif args.dataset[i] == 'MSP60k':
                            label_vec = torch.zeros(57, 768)
                        elif args.dataset[i] == 'DUKE':
                            label_vec = torch.zeros(36, 768)
                        elif args.dataset[i] == 'EventPAR':
                            label_vec = torch.zeros(50, 768)

                    if torch.cuda.is_available():
                        label_vec = label_vec.cuda()

                else:
                    # 原始多模态：使用真实的word_vec
                    label_vec = (cache['label_v'][i][0])[:attr_num[i]]

                cache['imgs'][i] = cache['imgs'][i][max_p:]
                cache['gt_labels'][i] = cache['gt_labels'][i][args.batchsize:]

                if args.dataset[i] == 'PA100k':
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss = criterions[i](train_logits, batch_gt_label) * lossrate[i]
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())

                elif args.dataset[i] == 'MSP60k':
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss = criterions[i](train_logits, batch_gt_label) * lossrate[i]
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())

                elif args.dataset[i] == 'DUKE':
                    H, W = batch_imgs.shape[-2], batch_imgs.shape[-1]
                    batch_imgs = batch_imgs.view(args.batchsize, N, -1, H, W)
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss = criterions[i](train_logits, batch_gt_label) * lossrate[i]
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())
                elif args.dataset[i] == 'EventPAR':
                    H, W = batch_imgs.shape[-2], batch_imgs.shape[-1]
                    batch_imgs = batch_imgs.view(args.batchsize, N, -1, H, W)
                    rgbs, events = batch_imgs[:, :N // 2, :, :, :], batch_imgs[:, N // 2:, :, :, :]
                    train_logits = model(rgbs, events, label_vec)
                    train_loss = criterions[i](train_logits, batch_gt_label) * lossrate[i]
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())
        log_interval = 2000
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/8:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')


    train_loss = loss_meter.avg


    gt_label = [np.concatenate(gt_list[i], axis=0) for i in range(len(args.dataset))]
    preds_probs = [np.concatenate(preds_probs[i], axis=0) for i in range(len(args.dataset))]

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs




def batch_trainer1(epoch, model, train_loader, criterions, optimizer, args, dicts):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = [[] for _ in range(len(args.dataset))]
    preds_probs = [[] for _ in range(len(args.dataset))]

    lr = optimizer.param_groups[0]['lr']
    # if args.accumulate:
    #     optimizer.zero_grad()
    #     train_loss_ = 0
    cache = {'imgs': [[] for _ in range(len(args.dataset))],
         'gt_labels': [[] for _ in range(len(args.dataset))],
         'label_v':  [[] for _ in range(len(args.dataset))],
         'dataset_ids': [[] for _ in range(len(args.dataset))],
         'imgidxs': [[] for _ in range(len(args.dataset))]}
    lossrate = []
    for i in range(len(args.dataset)):
        cache['label_v'][i].append(None)
        lossrate.append(1)
    if args.islossrate:
        lossrate = args.lossrate
    N_list, attr_num = train_loader.dataset.get_N()

    for step, (imgs, gt_labels, label_vs, dataset_ids, mask_id, imgidxs) in enumerate(train_loader):
        imgs,  gt_labels, label_vs = imgs.cuda(),  gt_labels.cuda(), label_vs.cuda()
        for i in range(len(args.dataset)):
            mask1 = mask_id == i
            mask2 = dataset_ids == i
            mask_list = mask2.tolist()
            if sum(mask2) == 0:
                continue
            imgs_i = imgs[mask1]
            gt_labels_i, label_vs_i = gt_labels[mask2], (label_vs[mask2])[0]
            imgidxs_i = [imgidxs[i] for i, keep in enumerate(mask_list) if keep]
            cache['imgs'][i].extend(imgs_i)
            cache['gt_labels'][i].extend(gt_labels_i)
            cache['label_v'][i][0] = label_vs_i
            cache['imgidxs'][i].extend(imgidxs_i)  # dataset_ids.sum()

            max_p = N_list[i] * args.batchsize
            N = N_list[i]
            batch_time = time.time()
            while len(cache['imgs'][i]) >= max_p:
                
                print(i)
                batch_imgs = torch.stack(cache['imgs'][i][:max_p])
                batch_gt_label = torch.stack(cache['gt_labels'][i][:args.batchsize])[:, :attr_num[i]]
                label_vec = (cache['label_v'][i][0])[:attr_num[i]]
                batch_imgidxs = cache['imgidxs'][i][:args.batchsize]
                cache['imgs'][i] = cache['imgs'][i][max_p:]
                cache['gt_labels'][i] = cache['gt_labels'][i][args.batchsize:]
                cache['imgidxs'][i] = cache['imgidxs'][i][args.batchsize:]
                if args.dataset[i] == 'PA100k':
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss, topk_indices = criterions[i](train_logits, batch_gt_label, 4) * lossrate[i]
                    selected_data_ids = [batch_imgidxs[i] for i in topk_indices.cpu().numpy()]
                    for x in selected_data_ids:
                        dicts[i][x] += 1
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())

                if args.dataset[i] == 'MSP60k':
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss, topk_indices = criterions[i](train_logits, batch_gt_label, 4) * lossrate[i]
                    selected_data_ids = [batch_imgidxs[i] for i in topk_indices.cpu().numpy()]
                    for x in selected_data_ids:
                        dicts[i][x] += 1
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())

                elif args.dataset[i] == 'DUKE':
                    H, W = batch_imgs.shape[-2], batch_imgs.shape[-1]
                    batch_imgs = batch_imgs.view(args.batchsize, N, -1, H, W)
                    train_logits = model(batch_imgs, None, label_vec)
                    train_loss, topk_indices = criterions[i](train_logits, batch_gt_label, 4) * lossrate[i]
                    selected_data_ids = [batch_imgidxs[i] for i in topk_indices.cpu().numpy()]
                    for x in selected_data_ids:
                        dicts[i][x] += 1
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())
                elif args.dataset[i] == 'EventPAR':
                    H, W = batch_imgs.shape[-2], batch_imgs.shape[-1]
                    batch_imgs = batch_imgs.view(args.batchsize, N, -1, H, W)
                    rgbs, events = batch_imgs[:, :N // 2, :, :, :], batch_imgs[:, N // 2:, :, :, :]
                    train_logits = model(rgbs, events, label_vec)
                    train_loss = criterions[i](train_logits, batch_gt_label) * lossrate[i]
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_meter.update(to_scalar(train_loss))
                    gt_list[i].append(batch_gt_label.cpu().numpy())
                    train_probs = torch.sigmoid(train_logits)
                    preds_probs[i].append(train_probs.detach().cpu().numpy())
        log_interval = 2000
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/8:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')

 

    train_loss = loss_meter.avg

    gt_label = [np.concatenate(gt_list[i], axis=0) for i in range(len(args.dataset))]
    preds_probs = [np.concatenate(preds_probs[i], axis=0) for i in range(len(args.dataset))]

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, dicts


def valid_trainer(model, valid_loader, criterion,args):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, label_v, _) in enumerate(valid_loader):
            
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            if args.visual_only:
                # 获取原始word_vec的形状
                original_word_vec = label_v[0]
                # 创建相同形状的零张量
                zero_word_vec = torch.zeros_like(original_word_vec).cuda()
                label_v = zero_word_vec
            else:
                # 多模态版本：使用原始的word_vec
                label_v = label_v[0].cuda()
            if isinstance(imgs, list):
                rgbs, events = imgs[0], imgs[1]
                rgbs, events = rgbs.cuda(), events.cuda()
                valid_logits = model(rgbs, events, label_v)
            else:
                imgs = imgs.cuda()
                valid_logits = model(imgs, None, label_v)
            
            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))


    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs

# def visual_only_valid_trainer(model, valid_loader, criterion):
#     model.eval()
#     loss_meter = AverageMeter()

#     preds_probs = []
#     gt_list = []
#     # 打印验证集信息（辅助调试）
    
#     # 核心：从MultiDataset中获取数据集名称（适配多数据集场景）
#     multi_valid_dataset = valid_loader.dataset  # 即前文的multi_valid_set（MultiDataset实例）
#     # 1. 获取所有包含的数据集名称（列表形式）
#     dataset_names = list(multi_valid_dataset.datasets.keys())  # 关键：访问MultiDataset的datasets字典
    
#     # 2. 定义各数据集的属性数量映射（统一管理，便于维护）
#     dataset_attr_num_map = {
#         'PA100k': 26,
#         'MSP60k': 57,
#         'DUKE': 36,
#         'EventPAR': 50
#     }
    
#     # 3. 遍历每个数据集，按名称匹配属性数量（支持多数据集验证）
#     for dataset_name in dataset_names:
#         # 获取当前数据集的属性数量（兜底默认值50）
#         num_attributes = dataset_attr_num_map.get(dataset_name, 50)
#         print(f"当前验证数据集: {dataset_name}, 属性数量: {num_attributes}")

#     with torch.no_grad():
#         for step, (imgs, gt_label, label_v, _) in enumerate(valid_loader):
#             # 处理标签
#             gt_label = gt_label.cuda()
#             gt_list.append(gt_label.cpu().numpy())
#             gt_label[gt_label == -1] = 0

#             # ========== 关键修改：创建占位符word_vec ==========
#             batch_size = imgs.shape[0] if not isinstance(imgs, list) else imgs[0].shape[0]
#             feature_dim = 768  # word_vec的特征维度

#             # 创建零张量作为占位符
#             # 形状: [num_attributes, feature_dim]
#             placeholder_word_vec = torch.zeros(num_attributes, feature_dim).cuda()
#             # ================================================

#             if isinstance(imgs, list):
#                 rgbs, events = imgs[0], imgs[1]
#                 rgbs, events = rgbs.cuda(), events.cuda()
#                 valid_logits = model(rgbs, events, placeholder_word_vec)
#             else:
#                 imgs = imgs.cuda()
#                 valid_logits = model(imgs, None, placeholder_word_vec)

#             valid_loss = criterion(valid_logits, gt_label)
#             valid_probs = torch.sigmoid(valid_logits)
#             preds_probs.append(valid_probs.cpu().numpy())
#             loss_meter.update(to_scalar(valid_loss))


#     valid_loss = loss_meter.avg
#     gt_label = np.concatenate(gt_list, axis=0)
#     preds_probs = np.concatenate(preds_probs, axis=0)

#     return valid_loss, gt_label, preds_probs
