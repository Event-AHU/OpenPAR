import time
from torch.nn import NLLLoss
import numpy as np
import torch
from tqdm import tqdm
from tools.utils import AverageMeter, to_scalar, time_str
def valid_trainer(model, valid_loader,vocab_attr,base_index2attr,attributes,args):
    model.eval()
    loss_meter = AverageMeter()
    loss_fn = NLLLoss(ignore_index=vocab_attr['<pad>'],reduction='mean')
    preds_probs = []
    gt_list = []
    image_pred_attr=[]
    image_gt_attr=[]
    image_names=[]
    count=0
    with torch.no_grad():
        for imgs, gt_label, imgname in tqdm(valid_loader):
            
            image_names.append(imgname[0])
            captions=get_gt_captions(gt_label,vocab_attr,base_index2attr,max_len=len(attributes))
            valid_loss=0
            imgs, gt_captions = imgs.cuda(), torch.Tensor(captions).cuda().detach().long()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())

            out_pred=model.beam_search_generate(imgs,max_length=len(attributes)+2,num_beams=args.beam_size,do_sample=args.do_sample,top_k=args.top_k,top_p=args.top_p,length_penalty=args.length_penalty)
            pred_attr=[]
            for elem in out_pred[0][1:]:
                if int(elem) > 2 :
                    pred_attr.append(vocab_attr.lookup_token(int(elem)))
            gt_attr=[]       
            for elem in gt_captions[0][1:]:
                if int(elem) > 2 :
                    gt_attr.append(vocab_attr.lookup_token(int(elem)))            
            image_pred_attr.append(pred_attr)
            image_gt_attr.append(gt_attr)
            probs=logit2prob(out_pred[:,1:],gt_label,vocab_attr,base_index2attr,'test',max_len=len(attributes)) 
            preds_probs.append(probs)


    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return  gt_label, preds_probs,0,image_names,image_pred_attr,image_gt_attr

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None :
            p.grad.data = p.grad.data.float()
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
            if elem > 2 :
                attr_index=get_key(base_index2attr,vocab_attr.lookup_token(int(elem)))[0]
                if attr_index<max_len:
                    probs[ss,attr_index]=1 
            
    return probs
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def advise_nllloss(out,gt_captions,criterion):
    loss=0
    for bl,bs in enumerate(gt_captions) :
        gt_line=gt_captions[bl].expand(gt_captions.shape[1],gt_captions.shape[1])
        out_line=torch.cat([out[bl][aa].expand(out.shape[1],out.shape[2]).unsqueeze(0) for aa,a_line in enumerate(bs)] ,dim=0)
        loss+=criterion(out_line.contiguous().view(-1, out.shape[-1]), gt_line.contiguous().view(-1))
    loss=loss/gt_captions.shape[0]
    return loss