import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
from torch.nn import NLLLoss
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tools.function import get_pedestrian_metrics
from tools.utils import AverageMeter, to_scalar, time_str
import pdb
import log_utils

def batch_trainer(epoch, model, train_loader, criterion, optimizer,vocab_attr,base_index2attr,attributes,args):
    
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()
    #prompt_loss_meter= AverageMeter()
    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    loss_fn = NLLLoss(ignore_index=vocab_attr['<pad>'],reduction='mean')
    lr = optimizer.param_groups[0]['lr']
    metric_logger = log_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)    
    for it, (imgs, gt_label, imgname) in enumerate(metric_logger.log_every(train_loader, int(batch_num/10), header)):
        captions=get_gt_captions(gt_label, vocab_attr, base_index2attr, max_len=20)
        imgs, gt_captions = imgs.cuda(), torch.Tensor(captions).cuda().long()
        out,train_logits = model(imgs,input=gt_captions)
        train_loss = loss_fn(out[:,:-1,:].contiguous().view(-1, out.shape[-1]), gt_captions[:,1:].contiguous().view(-1))
        optimizer.zero_grad()
        train_loss.backward()#反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        probs=logit2prob(train_logits[:,:-1],gt_label,vocab_attr,base_index2attr,'train',max_len=20) 
        preds_probs.append(probs)
        metric_logger.update(train_loss=train_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger) 
    train_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')
    return train_loss, gt_label, preds_probs


def valid_trainer(model, valid_loader,vocab_attr,base_index2attr,attributes,args):
    model.eval()
    loss_meter = AverageMeter()
    loss_fn = NLLLoss(ignore_index=vocab_attr['<pad>'],reduction='mean')
    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(valid_loader):
            captions = get_gt_captions(gt_label,vocab_attr,base_index2attr,max_len=len(attributes))
            valid_loss=0
            imgs, gt_captions = imgs.cuda(), torch.Tensor(captions).cuda().detach().long()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            out_pred=model.beam_search_generate(imgs,max_length=20,num_beams=args.beam_size)
            probs=logit2prob(out_pred[:,1:],gt_label,vocab_attr,base_index2attr,'test',max_len=len(attributes)) 
            preds_probs.append(probs)
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return  gt_label, preds_probs,valid_loss
            
def get_gt_captions(gt_label,vocab_attr,base_index2attr,max_len=51):

    gt_captions=[[vocab_attr['<bos>']] for i in range(gt_label.shape[0])]
    for bb,batchs in enumerate(gt_label) :
        for count,elem in enumerate(batchs):
            if elem :
               gt_captions[bb].append(vocab_attr[base_index2attr[count]])
        if  len(gt_captions[bb])<=max_len :
            for _ in range(max_len-len(gt_captions[bb])+1) :
                gt_captions[bb].append(vocab_attr['<pad>'])
        if len(gt_captions[bb])!=max_len+1 :
            raise Exception(f'长度为{len(gt_captions[bb])}')
    return gt_captions
def logit2prob(input_probs: torch.Tensor,gt_label,vocab_attr,base_index2attr,state,max_len=51) :
    probs=np.zeros((gt_label.shape[0],gt_label.shape[1]))
    if state=='train' :
        input_probs=torch.max(input_probs,2)[1].cpu().detach().numpy()#输出每个位置预测概率最大的索引
    for ss,batchs in enumerate(input_probs) :#将索引对应位置设值
        for elem in batchs :
            if elem > 3 :
                attr_index=get_key(base_index2attr,vocab_attr.lookup_token(int(elem)))[0]
                if attr_index<max_len:
                    probs[ss,attr_index]=1 
            
    return probs
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]
