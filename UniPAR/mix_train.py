import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, mix_batch_trainer
from config import argument_parser
from dataset.AttrDataset import get_mix_multi_dataset, custom_collate
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
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)

    multi_train_set, multi_valid_set, criterion_dict = get_mix_multi_dataset(args)

    train_loader = DataLoader(
        dataset=multi_train_set,
        batch_size=args.batchsize * 2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )

    valid_loader = DataLoader(
        dataset=multi_valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.visual_only:
        model = VisualOnlyTransformerClassifier(dim=768, args=args)
    else:
        model = TransformerClassifier(dim=768, args=args)
    # checkpoint_path = "/wangx/DATA/Code/yanzikang/PAR/code/logs/multiDataset/2025-10-09_17_42_20/ckpt_2025-10-11_00_55_36_20.pth"
#     checkpoint_path ="/wangx/DATA/Code/xujiarui/code/logs/multiDataset/2025-12-15_13_28_11/ckpt_2025-12-17_19_52_52_40.pth"
#     state_dict = torch.load(checkpoint_path, weights_only=False)
#     print(checkpoint_path)
#     model.load_state_dict(state_dict['state_dicts'],strict=False)

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
    criterions = []
    for dataset in args.dataset:
        criterions.append(criterion_dict[dataset])

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch, lr=args.lr, warmup_t=5)
    for i in range(1, epoch+1):

        scheduler.step(i)
        train_loss, train_gt, train_probs = mix_batch_trainer(
                epoch=i,
                model=model,
                train_loader=train_loader,
                criterions=criterions,
                optimizer=optimizer,
                args=args
            )
        
        for idx, dataset in enumerate(args.dataset):
            valid_loader.dataset.init_set(dataset)
            criterion = criterion_dict[dataset]
            # 使用原始验证函数
            valid_loss, valid_gt, valid_probs = valid_trainer(
                    model=model,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    args=args
            )

            train_result = get_pedestrian_metrics(train_gt[idx], train_probs[idx])

            print(f'Evaluation on train set:{dataset}, \n',
                  'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                      train_result.ma, np.mean(train_result.label_pos_recall), np.mean(train_result.label_neg_recall)),
                  'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                      train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                      train_result.instance_f1))
            
            valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

            print(f'Evaluation on test set:{dataset}, \n',
                  'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                      valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
                  'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                      valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                      valid_result.instance_f1))
        
        print('-' * 60)
        if i % args.epoch_save_ckpt == 0:
            save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)
        
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()