import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn,optim
from eval_batch import valid_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.test_base import *
from tools.function import get_pedestrian_metrics,get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
from CLIP.clip import clip
from CLIP.clip.model import *
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
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    print("start loading model")
    checkpoint = torch.load(args.dir)
    ViT_model=build_model(checkpoint['ViT_model'])
    model = TransformerClassifier(ViT_model,train_set.attr_num,train_set.attributes)
    #CUDA_VISIBLE_DEVICES=0 python eval.py RAPV1 --checkpoint --dir ./logs/RAPV1/2023-10-17_19_36_32/epoch23.pth --use_div --use_vismask --vis_prompt 50 --use_GL --use_textprompt --use_mm_former 
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
        ViT_model=ViT_model.cuda()
    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    trainer(model=model,
            valid_loader=valid_loader,
            ViT_model=ViT_model,
            criterion=criterion,
            args=args)
attributes= [
    'Female',
    'AgeLess16','Age17-30','Age31-45',
    'BodyFat','BodyNormal','BodyThin','Customer','Clerk'
    'hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses','hs-Muffler',
    'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve',
    'lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers', 
    'shoes-Leather','shoes-Sport','shoes-Boots','shoes-Cloth','shoes-Casual',
    'attach-Backpack','attach-SingleShoulderBag','attach-HandBag','attach-Box','attach-PlasticBag','attach-PaperBag','attach-HandTrunk','attach-Other',
    'action-Calling','action-Talking','action-Gathering','action-Holding','action-Pusing','action-Pulling','action-CarrybyArm','action-CarrybyHand'
]
def trainer(model,valid_loader,ViT_model,criterion,args):
    for num in attributes:
        formatted_num = f"{num}"
        print(formatted_num,end=',') 
        
    start=time.time()
    valid_loss, valid_gt, valid_probs,image_names = valid_trainer(
        model=model,
        ViT_model=ViT_model,
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
    ma = []
    acc = []
    f1 = []
    valid_probs = valid_probs > 0.45
    pred_attrs=[[] for _ in range(len(image_names))]
    gt_attrs=[[] for _ in range(len(image_names))]
    for pidx in range(len(image_names)):
        for aidx in range(len(attributes)):
            if valid_probs[pidx][aidx] : 
                pred_attrs[pidx].append(attributes[aidx])
            if valid_gt[pidx][aidx] : 
                gt_attrs[pidx].append(attributes[aidx])
    # 打开一个文本文件以写入模式
    with open('preds_img_attrs.txt', 'w') as file:
        # 遍历字典的键值对，按行写入文本文件
        for pidx in range(len(image_names)):
            file.write(f'{image_names[pidx]}: {pred_attrs[pidx]}\n')   
    # 打开一个文本文件以写入模式
    with open('gt_img_attrs.txt', 'w') as file:
        # 遍历字典的键值对，按行写入文本文件
        for pidx in range(len(image_names)):
            file.write(f'{image_names[pidx]}: {gt_attrs[pidx]}\n')              
    end=time.time()
    total=end-start 
    print(f'The time taken for the test epoch is:{total}')                 

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)