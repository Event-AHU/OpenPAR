import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from spikingjelly.clock_driven import functional
from tools.utils import AverageMeter, to_scalar, time_str
import torch.nn.functional as F
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']

    for step, (imgs, gt_label, imgname, label_v, label_n) in enumerate(train_loader):
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        label_v = label_v[0].cuda()
        
        T_train_logits,T_img_features,text_features,S_train_logits, S_img_features = model(imgs, label_v, gt_label)
        cosine_sim1 = F.cosine_similarity(text_features.unsqueeze(2), S_img_features.unsqueeze(1), dim=-1) #(B,26,128)

        cosine_sim2 = F.cosine_similarity(text_features.unsqueeze(2), T_img_features.unsqueeze(1), dim=-1) #(B,26,128)

        kl_loss_1 = F.kl_div(F.log_softmax(cosine_sim1, dim=-1), F.softmax(cosine_sim2, dim=-1), reduction='batchmean')

        kl_loss_2 = F.kl_div(F.log_softmax(S_train_logits/args.temp, dim=-1), F.softmax(T_train_logits/args.temp, dim=-1), reduction='batchmean')*(args.temp*args.temp)

        ces_loss = criterion(S_train_logits, gt_label) 

        if args.only_feats_kl and args.only_logits_kl:

            train_loss = 0.6* ces_loss + 100 * kl_loss_1 + kl_loss_2

        elif args.only_feats_kl:
           
            train_loss =0.8 * ces_loss + kl_loss_1 *100

        elif args.only_logits_kl:
            
            train_loss = 0.8 * ces_loss + kl_loss_2 
        else:
            train_loss=ces_loss            

        train_loss.backward()
        
        optimizer.step()
        functional.reset_net(model)
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))
        
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(S_train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 200
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname, label_v, label_n) in enumerate(valid_loader):
            # if step==2:
            #     break
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            label_v = label_v[0].cuda()
            T_train_logits,T_img_features,text_features,S_train_logits, S_img_features = model(imgs, label_v)
            functional.reset_net(model)
            valid_loss = criterion(S_train_logits, gt_label)
            valid_probs = torch.sigmoid(S_train_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
