import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
import torchvision
import utils
import os
import time
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import TransformerClassifier as Trans
from models.base_block_attack import TransformerClassifier as Transattack
from tools.function import get_pedestrian_metrics,count_parameters
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
from clip import clip
from clip.model import *
set_seed(605)
device = "cuda"
# clip_model, ViT_preprocess = clip.load("ViT-L/14", device=device,download_root='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/kongweizhe/PromptPAR/checkpoints')


imagenet_mean = torch.Tensor([0.5, 0.5, 0.5]).to(device)
imagenet_std = torch.Tensor([0.5, 0.5, 0.5]).to(device)

def main(args):
    start_time = time_str()
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
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=valid_tsfm)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=False,
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


    #加载原有模型参数
    if(args.dataset == 'PA100k'):
        # checkpoint = torch.load('logs/PA100k/2025-03-30_22_57_04/best_f1.pth')
        checkpoint = torch.load('logs/PA100k/2025-04-17_17_27_16/min_f1.pth')
        clip_model = build_model(checkpoint['clip_model']).cuda()
    if(args.dataset == 'MSPAR'):
        checkpoint = torch.load('logs/MSPAR/2025-04-15_15_54_05/min_f1.pth')
        clip_model = build_model(checkpoint['clip_model']).cuda()
    if(args.dataset == 'RAPV2'):
        checkpoint = torch.load('logs/RAPV2/2025-03-30_23_20_42/best_f1.pth')
        clip_model = build_model(checkpoint['clip_model']).cuda()
    if(args.dataset == 'PETA'):
        checkpoint = torch.load('logs/PETA/noLimit_SP_LP/epoch75.pth')
        # checkpoint = torch.load('logs/PETA/2024-11-22_12_04_09/epoch44.pth')
        # breakpoint()
        clip_model = build_model(checkpoint['clip_model']).cuda()
    model = Transattack(clip_model, train_set.attr_num, train_set.attributes)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)


    #加载噪声参数
    attack_checkpoint = torch.load('logs/PETA/noLimit_SP_LP/epoch75.pth')
    # attack_checkpoint = torch.load('logs/PETA/2024-11-22_12_04_09/epoch44.pth')
    attack_model = Transattack(clip_model, 35, train_set.attributes)
    if torch.cuda.is_available():
        attack_model = attack_model.cuda()
    attack_model.load_state_dict(attack_checkpoint["model_state_dict"], strict=False)
    epoch = attack_checkpoint["epoch"]
    print(epoch)

    mm_params=[]
    #Freeze parameters other than category headers and prompts
    for name, param in model.named_parameters():
        param.requires_grad = False

    
    clip_params=[]
    for name, param in clip_model.named_parameters():
        param.requires_grad = False
    
    attack_params=[]
    for name, param in attack_model.named_parameters():
        param.requires_grad = False


    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

    noisy = attack_model.attack_noisy.data.cuda()

    model.attack_noisy.data = noisy
    # for epoch in range(args.epoch):
    model.eval()
    tq = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    preds_probs = []
    gt_list = []
    for batch_idx, (images, labels, images_name) in tq:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits,_ = model(images, clip_model)
        if (batch_idx==142):
            # breakpoint()
            images = F.interpolate(images[39].unsqueeze(0), size=(256, 128), mode="bilinear", align_corners=False)
            image = images.squeeze()
            image = torch.einsum("chw->hwc",images.squeeze())
            image = image * imagenet_std + imagenet_mean
            image = torch.einsum("hwc->chw",image)
            torchvision.utils.save_image(image, "output_image.jpg")
            breakpoint()
        
        # logits,_ = model(images + noisy, clip_model)
        # breakpoint()
        loss = criterion(logits, labels)
        optimizer.step()

        gt_list.append(labels.cpu().numpy())
        probs = torch.sigmoid(logits)
        
        preds_probs.append(probs.cpu().numpy())
        result = get_pedestrian_metrics(labels.cpu().numpy(), probs.cpu().numpy())
        tq.set_description('loss{:.4f}, ' \
                    'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f}, ' \
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(loss,
                    result.ma, np.mean(result.label_pos_recall), np.mean(result.label_neg_recall),
                    result.instance_acc, result.instance_prec, result.instance_recall, result.instance_f1))
        
    lr_scheduler.step()
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    result = get_pedestrian_metrics(gt_label, preds_probs)
    print('loss{:.4f}, ' \
                'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f}, ' \
                'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(loss,
                result.ma, np.mean(result.label_pos_recall), np.mean(result.label_neg_recall),
                result.instance_acc, result.instance_prec, result.instance_recall, result.instance_f1))


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
