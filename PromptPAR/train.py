import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,count_parameters
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
import torch.optim as optim
from CLIP.clip import clip
from CLIP.clip.model import *
set_seed(605)
device = "cuda"
ViT_model, ViT_preprocess = clip.load("ViT-L/14", device=device,download_root='/data/jinjiandong') #选择image特征提取器

def main(args):
    start_time=time_str()
    print(f'start_time is {start_time}')
    log_dir = os.path.join('logs', args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, start_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')
    train_tsfm, valid_tsfm = get_transform(args) 
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) 
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize, 
        shuffle=True,
        num_workers=8,  
        pin_memory=True,  
    )
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    labels = train_set.label
    sample_weight = labels.mean(0)
    
    model = TransformerClassifier(ViT_model,train_set.attr_num,train_set.attributes)
    if torch.cuda.is_available():
        model = model.cuda()
    mm_params=[]
    #Freeze parameters other than category headers and prompts
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in args.mmformer_update_parameters):
            mm_params+= [{
            "params": [param],
            "lr": args.lr,
            "weight_decay": args.weight_decay
            }]
        else:
            param.requires_grad = False
    clip_params=[]
    for name, param in ViT_model.named_parameters():
        if any(keyword in name for keyword in args.clip_update_parameters):
            print(name, param.requires_grad)
            clip_params+= [{
            "params": [param],
            "lr": args.clip_lr,
            "weight_decay": args.clip_weight_decay
            }]
        else:
            param.requires_grad = False

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    epoch_num = args.epoch                                                                                                                                                                                            
    count_parameters(model,ViT_model,args.mmformer_update_parameters,args.clip_update_parameters)
                
    prompt_optimizer = optim.SGD(clip_params, args.clip_lr, momentum=0.9, weight_decay=args.clip_weight_decay)
    prompt_scheduler = make_scheduler(prompt_optimizer,num_epochs=epoch_num,lr=args.clip_lr,warmup_t=10)
    
    optimizer = optim.SGD(mm_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=args.lr, warmup_t=5)

    trainer(epoch=epoch_num,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            prompt_scheduler=prompt_scheduler,
            prompt_optimizer=prompt_optimizer,
            args=args,
            path=log_dir)
    
def trainer(epoch, model,ViT_model, train_loader, valid_loader, criterion, optimizer, scheduler,prompt_scheduler,prompt_optimizer,args,path):
    max_ma,max_acc,max_f1,=0,0,0
    start=time.time()
    for i in range(1, epoch+1):
        scheduler.step(i)
        prompt_scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            prompt_optimizer=prompt_optimizer,
            args=args
        )
        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            ViT_model=ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
            args=args
        )
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
        print(f'Evaluation on test set, valid_loss:{valid_loss:.4f}\n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))  
                       
        if valid_result.ma>=max_ma :
            max_ma=valid_result.ma
        if valid_result.instance_acc>=max_acc :
            max_acc=valid_result.instance_acc
        if valid_result.instance_f1>=max_f1 :
            max_f1=valid_result.instance_f1
            
        print('-' * 60)
        if i % args.save_freq == 0:
            torch.save({
                        'epoch': i,
                        'optimizer':optimizer.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'ViT_model' : ViT_model.state_dict(),
                        'result',valid_result
                        }, os.path.join(path, f"epoch{i}.pth"))    
        if i%5==0:
            end=time.time()
            total=end-start
            print(f'The time taken for the last {i} epoch is:{total}')
            print(f'max_ma:{max_ma:.4f},max_acc:{max_acc:.4f},max_f1:{max_f1:.4f}') 
            
              


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)