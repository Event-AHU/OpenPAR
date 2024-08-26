import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn,optim
from batch_engine import valid_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
from clip import clip
from clip.model import *

set_seed(605)
device = "cuda"
def main(args):

    if args.checkpoint==False :
        print(time_str())
        pprint.pprint(OrderedDict(args.__dict__))
        print('-' * 60)
        print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args) 
    train_tsfm, valid_tsfm = get_transform(args) 
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) 
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 
    labels = train_set.label
    sample_weight = labels.mean(0)    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    print("start loading model")
    checkpoint = torch.load(args.dir)
    clip_model = build_model(checkpoint['ViT_model'])
    
    model = TransformerClassifier(clip_model,train_set.attr_num,train_set.attributes)
    # 
    #CUDA_VISIBLE_DEVICES=4 python eval.py RAPV1 --checkpoint --dir /data1/Code/jinjiandong/OpenPAR-main/PromptPAR/logs/PETA/2024-05-23_14_59_25/epoch21.pth
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
        clip_model=clip_model.cuda()
    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)

    attributes = train_set.attributes
    
    for num in attributes:
        formatted_num = f"{num}"
        print(formatted_num,end=',') 
        
    start=time.time()
    valid_loss, valid_gt, valid_probs = valid_trainer(
        model=model,
        clip_model=clip_model,
        valid_loader=valid_loader,
        criterion=criterion,
        args=args
    )
    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
    print('valid_loss: {:.4f}'.format(valid_loss))
    print('ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                valid_result.instance_f1))     
     
    end=time.time()
    total=end-start 
    print(f'The time taken for the test epoch is:{total}')                 

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
