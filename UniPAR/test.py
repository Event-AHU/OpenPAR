import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import get_multi_dataset
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

set_seed(605)

def main(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # 将--gpus参数映射到环境变量
        print(f"Using GPU(s): {args.gpus}")
    start_time=time_str()
    print(f'start_time is {start_time}')
    log_dir = os.path.join('logs', args.save_place)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, start_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_test_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)

    multi_train_set, multi_valid_set, criterion_dict = get_multi_dataset(args)

    train_loader = DataLoader(
        dataset=multi_train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=multi_valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    model = TransformerClassifier(dim=768,args=args)

    # checkpoint_path = "/wangx/DATA/Code/partest/code/logs/multiDataset/2025-10-29_09_57_53/ckpt_2025-11-02_09_28_47_50.pth"
    # checkpoint_path = "/wangx/DATA/Code/yanzikang/PAR/code/logs/multiDataset/2025-10-09_17_42_20/ckpt_2025-10-11_00_55_36_20.pth"
    checkpoint_path = "/wangx/DATA/Code/partest/code/logs/multiDataset/ckpt_2025-11-12_17_05_12_50.pth"
    state_dict = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(state_dict['state_dicts'])

    if torch.cuda.is_available():
        model = model.cuda()


    trainer(epoch=args.epoch,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion_dict=criterion_dict,
            path=log_dir,
            args=args)
    
def trainer(epoch, model, train_loader, valid_loader, criterion_dict, path, args):
    for dataset in args.dataset:
        print(f'================================', dataset, '================================', '\n')
        torch.cuda.empty_cache()
        optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = create_scheduler(optimizer, num_epochs=args.epoch, lr=args.lr, warmup_t=5)
        criterion = criterion_dict[dataset]
        train_loader.dataset.init_set(dataset)
        valid_loader.dataset.init_set(dataset)
        # model = vit_load(model, dataset)
        # model = patch_embed_load(model, dataset)
        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

            
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print('-' * 60)

def patch_embed_load(model, dataset):
    if dataset == 'PA100k':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-23_03_37_18_20.pth'
        state_dict = torch.load(checkpoint_path)
        PE_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                PE_dict[k] = v
        model.load_state_dict(PE_dict, strict=False)
    if dataset == 'DUKE':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-23_06_57_01_100.pth'
        state_dict = torch.load(checkpoint_path)
        PE_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                PE_dict[k] = v
        model.load_state_dict(PE_dict, strict=False)
    if dataset == 'EventPAR':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-25_13_59_09_20.pth'
        state_dict = torch.load(checkpoint_path)
        PE_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                PE_dict[k] = v
        model.load_state_dict(PE_dict, strict=False)
    return model

def vit_load(model, dataset):
    if dataset == 'PA100k':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-23_03_37_18_20.pth'
        state_dict = torch.load(checkpoint_path)
        Vit_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                continue
            Vit_dict[k] = v
        model.load_state_dict(Vit_dict, strict=False)
    if dataset == 'DUKE':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-23_06_57_01_100.pth'
        state_dict = torch.load(checkpoint_path)
        Vit_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                continue
            Vit_dict[k] = v
        model.load_state_dict(Vit_dict, strict=False)
    if dataset == 'EventPAR':
        checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-06-22_23_06_49/ckpt_2025-06-25_13_59_09_20.pth'
        state_dict = torch.load(checkpoint_path)
        Vit_dict = OrderedDict()
        for k, v in state_dict['state_dicts'].items():
            if 'patch_embed' in k and 'vit' not in k:
                continue
            Vit_dict[k] = v
        model.load_state_dict(Vit_dict, strict=False)
    return model

            

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()