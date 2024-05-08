import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn,optim
torch.autograd.set_detect_anomaly(True)
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from models.sidenet import *
from tools.function import get_pedestrian_metrics,get_signle_metrics, simple_par_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler

from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter
set_seed(12240321)
device = "cuda" if torch.cuda.is_available() else "cpu"

ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian') 

def main(args):
    start_time=time_str()
    print(f'start_time is {start_time}')
    log_dir = os.path.join('logs', args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, start_time)
    tb_writer = SummaryWriter(r'/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zhuqian/Cross_ViT/exp')
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

    train_set = MultiModalAttrDataset(args=args, split=args.train_split , transform=train_tsfm) 
    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=args.batchsize, 
        shuffle=True,
        num_workers=8,
        pin_memory=True, 
    )
    
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split , transform=valid_tsfm) 
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    labels = train_set.label
    sample_weight = labels.mean(0) 
    model = TransformerClassifier(ViT_model, train_set.attr_num, attr_words=train_set.attributes)
    
    if torch.cuda.is_available():
        model = model.cuda()
    for name, param in model.named_parameters():
        if 'clip_visual_extractor' in name:
            param.requires_grad = False
    
    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    lr = args.lr
    epoch_num = args.epoch
    start_epoch=1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)
    count_parameters(model, ViT_model.transformer)

    best_metric, epoch = trainer(args=args,
                                 epoch=epoch_num,
                                 model=model,
                                 ViT_model=ViT_model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir,
                                 tb_writer=tb_writer,
                                 start_epoch=start_epoch)
    
def trainer(args,epoch, model,ViT_model, train_loader, valid_loader, criterion, optimizer, scheduler,path,tb_writer,start_epoch):
    max_ma,max_acc,max_f1,=0,0,0
    start=time.time()
    valid_name_buf = []
    for i in range(start_epoch, epoch+1):
        scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )
        valid_loss, valid_gt, valid_probs = valid_trainer(
            epoch=epoch,
            model=model,
            ViT_model=ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        
        if args.dataset =='MARS' : 
            #MARS
            index_list=[0,1,2,3,4,5,6,7,8,9,15,20,29,39,43]
            group="top length, bottom type, shoulder bag, backpack, hat, hand bag, hair, gender, bottom length, pose, motion, top color, bottom color, age"
        else:
            #DUKE
            index_list=[0,1,2,3,4,5,6,7,8,14,19,28,36]
            group="backpack, shoulder bag, hand bag, boots, gender, hat, shoes, top length, pose, motion, top color, bottom color"
        group_f1=[]
        group_acc=[]
        group_prec=[]
        group_recall=[]
        for idx in range(len(index_list)-1):
            if index_list[idx+1]-index_list[idx] >1 :
                result=simple_par_metrics(valid_gt[:,index_list[idx]:index_list[idx+1]], valid_probs[:,index_list[idx]:index_list[idx+1]])
            elif idx < 9  :
                result=simple_par_metrics(valid_gt[:,index_list[idx]], valid_probs[:,index_list[idx]],signle=True)
            group_f1.append(result.f1) 
            group_acc.append(result.acc)  
            group_prec.append(result.prec)
            group_recall.append(result.recall)   
        average_instance_f1 = np.mean(group_f1)

        average_acc = np.mean(group_acc)
        average_prec = np.mean(group_prec)    
        average_recall = np.mean(group_recall)

        print(f'{time_str()}Evaluation on test set, valid_loss:{valid_loss:.4f}\n',
              f"Acc :{group} \n",','.join(str(elem)[:6] for elem in group_acc),'\n',
              f"Prec :",','.join(str(elem)[:6] for elem in group_prec),'\n',
              f"Recall :",','.join(str(elem)[:6] for elem in group_recall),'\n',
              f"F1 :{group}  \n",','.join(str(elem)[:6] for elem in group_f1),'\n',
              'average_acc: {:.4f},average_prec: {:.4f},average_recall: {:.4f},average_f1: {:.4f}'.format(average_acc, average_prec, average_recall, average_instance_f1)                 
                )
        print('-' * 60)        

        tb_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i )  
        #tb_writer.add_scalar("train_loss", train_loss, i )  
        tb_writer.add_scalar("valid_loss", valid_loss, i )  
        tb_writer.add_scalar("valid_F1", average_instance_f1, i )
        if i % args.epoch_save_ckpt == 0:
            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        }, os.path.join(path, f"epoch{i}.pth"))              
        
def count_parameters(model, ViT_model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in ViT_model.parameters())
    selected_params1 = []
    selected_params2 = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            selected_params1.append(param)
      
    selected_params_count1 = sum(p.numel() for p in selected_params1)
    trainable_percentage = ((selected_params_count1) / total_params) * 100 if total_params > 0 else 0
    print(f"trainable params: {(selected_params_count1)} || all params: {total_params} || trainable%: {trainable_percentage:.12f}")
       
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

   