import os
import os
import pprint
from collections import OrderedDict, defaultdict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
# from models.hop_block import *
from tools.function import get_pedestrian_metrics,simple_par_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
set_seed(605)

def main(args):
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    

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
    select_gpus(args.gpus)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        
    )

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

   
    model = TransformerClassifier(train_set.attr_num)
    
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    lr = args.lr
    epoch_num = args.epoch

   
    if args.optim == 'SGD' :
        optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
        print('The optimizer used SGD')
    elif args.optim == 'AdamW' :
        optimizer = optim.AdamW(model.parameters(),lr=lr, weight_decay=args.weight_decay)
        print('The optimizer used AdamW')
    else :
        optimizer = optim.Adam([{'params': model.decoder.parameters(), 'lr': args.lr},{'params': model.ViT_model.parameters(), 'lr': args.lr}])
        print('The optimizer used Adam')
   
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)

    best_metric, epoch = trainer(epoch=epoch_num,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir)
    
def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, scheduler, path):
    start=time.time()
    max_ma,max_acc,max_f1,=0,0,0
    for i in range(1, epoch+1):
        scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        
        train_result = get_pedestrian_metrics(train_gt, train_probs)
        
        print(f'{time_str()} on train set:\n',
              'ma: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(
                  train_result.ma, train_result.instance_acc, train_result.instance_f1))  
        
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
       
        print(f'{time_str()} on Evalution set:\n',
                'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                    valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
                'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                    valid_result.instance_f1))                 
        print('-' * 60)   

        if i % args.epoch_save_ckpt == 0:
            save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)
        
        if valid_result.ma>=max_ma :
            max_ma=valid_result.ma
        if valid_result.instance_acc>=max_acc :
            max_acc=valid_result.instance_acc
        if valid_result.instance_f1>=max_f1 :
            max_f1=valid_result.instance_f1

        if i%5==0:
            end=time.time()
            total=end-start
            print(f'The time taken for the last {i} epoch is:{total}')
            print(f'max_ma:{max_ma:.4f},max_acc:{max_acc:.4f},max_f1:{max_f1:.4f}')

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

   