import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, mix_batch_trainer, batch_trainer1
from config import argument_parser
from dataset.AttrDataset import get_mix_multi_dataset, custom_collate, get_mix_balance_dataset
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

set_seed(605)

def main(args):
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

    model = TransformerClassifier(args=args)

    checkpoint_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/sunminhao/VTB-main/logs/multiDataset/2025-07-03_18_34_48/ckpt_2025-07-07_13_43_52_40.pth'
    state_dict = torch.load(checkpoint_path)
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
    criterions = []
    for dataset in args.dataset:
        criterions.append(criterion_dict[dataset])

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=40, lr=args.lr, warmup_t=5)
    for i in range(1, 40+1):
        scheduler.step(i)
    dicts = []
    for _ in range(len(args.dataset)):
        dicts.append(defaultdict(int))
    
    for i in range(41, 46):

        scheduler.step(i)
        train_loss, train_gt, train_probs, dicts = batch_trainer1(
                epoch=i,
                model=model,
                train_loader=train_loader,
                criterions=criterions,
                optimizer=optimizer,
                args=args,
                dicts=dicts
            )
        
        for idx, dataset in enumerate(args.dataset):
            valid_loader.dataset.init_set(dataset)
            criterion = criterion_dict[dataset]
            valid_loss, valid_gt, valid_probs = valid_trainer(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
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
        
        for dict_i in dicts:
            cnt1, cnt2, cnt3 = 0, 0, 0
            for key, value in dict_i.items():
                cnt1 += 1
                cnt2 += value
            for k in range(1, i - 40 + 1):
                cnt3 = 0
                for key, value in dict_i.items():
                    if value >= k:
                        cnt3 += 1
                print(f"{k} {cnt3} {cnt1}") 
        
        print('-' * 60)
        if i % args.epoch_save_ckpt == 0:
            save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)

    imgidx0, imgidx1 = [], []
    for idx, cnt in dicts[0].items():
        if cnt >= 5:
            imgidx0.append(idx)

    for idx, cnt in dicts[1].items():
        if cnt >= 5:
            imgidx1.append(idx)
    # 重建数据集
    multi_train_set, _, criterion_dict = get_mix_balance_dataset(args, imgidx_PA100k=imgidx0, imgidx_DUKE=imgidx1)
    train_loader = DataLoader(
        dataset=multi_train_set,
        batch_size=args.batchsize * 2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    criterions = []
    for dataset in args.dataset:
        criterions.append(criterion_dict[dataset])

    for i in range(46, epoch + 1):
        scheduler.step(i)
        train_loss, train_gt, train_probs = mix_batch_trainer(
                epoch=i,
                model=model,
                train_loader=train_loader,
                criterions=criterions,
                optimizer=optimizer,
                args=args,
            )
        
        for idx, dataset in enumerate(args.dataset):
            valid_loader.dataset.init_set(dataset)
            criterion = criterion_dict[dataset]
            valid_loss, valid_gt, valid_probs = valid_trainer(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
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

# def trainer(epoch, model, train_loader, valid_loader, criterion_dict, path, args):
#     criterions = []
#     for dataset in args.dataset:
#         criterions.append(criterion_dict[dataset])

#     optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = create_scheduler(optimizer, num_epochs=epoch, lr=args.lr, warmup_t=5)
#     for i in range(1, 41):
#         scheduler.step(i)
#         train_loss, train_gt, train_probs = mix_batch_trainer(
#                 epoch=i,
#                 model=model,
#                 train_loader=train_loader,
#                 criterions=criterions,
#                 optimizer=optimizer,
#                 args=args
#             )
        
#         for idx, dataset in enumerate(args.dataset):
#             valid_loader.dataset.init_set(dataset)
#             criterion = criterion_dict[dataset]
#             valid_loss, valid_gt, valid_probs = valid_trainer(
#                 model=model,
#                 valid_loader=valid_loader,
#                 criterion=criterion,
#             )
#             train_result = get_pedestrian_metrics(train_gt[idx], train_probs[idx])

#             print(f'Evaluation on train set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       train_result.ma, np.mean(train_result.label_pos_recall), np.mean(train_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
#                       train_result.instance_f1))
            
#             valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

#             print(f'Evaluation on test set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
#                       valid_result.instance_f1))
        
#         print('-' * 60)
#         if i % args.epoch_save_ckpt == 0:
#             save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)
    
#     dicts = []
#     for _ in range(len(args.dataset)):
#         dicts.append(defaultdict(int))
    
#     for i in range(41, 46):

#         scheduler.step(i)
#         train_loss, train_gt, train_probs, dicts = batch_trainer1(
#                 epoch=i,
#                 model=model,
#                 train_loader=train_loader,
#                 criterions=criterions,
#                 optimizer=optimizer,
#                 args=args,
#                 dicts=dicts
#             )
        
#         for idx, dataset in enumerate(args.dataset):
#             valid_loader.dataset.init_set(dataset)
#             criterion = criterion_dict[dataset]
#             valid_loss, valid_gt, valid_probs = valid_trainer(
#                 model=model,
#                 valid_loader=valid_loader,
#                 criterion=criterion,
#             )
#             train_result = get_pedestrian_metrics(train_gt[idx], train_probs[idx])

#             print(f'Evaluation on train set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       train_result.ma, np.mean(train_result.label_pos_recall), np.mean(train_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
#                       train_result.instance_f1))
            
#             valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

#             print(f'Evaluation on test set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
#                       valid_result.instance_f1))
        
#         for dict_i in dicts:
#             cnt1, cnt2, cnt3 = 0, 0, 0
#             for key, value in dict_i.items():
#                 cnt1 += 1
#                 cnt2 += value
#             for k in range(1, i - 40 + 1):
#                 cnt3 = 0
#                 for key, value in dict_i.items():
#                     if value >= k:
#                         cnt3 += 1
#                 print(f"{k} {cnt3} {cnt1}") 
        
#         print('-' * 60)
#         if i % args.epoch_save_ckpt == 0:
#             save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)

#     imgidx0, imgidx1 = [], []
#     for idx, cnt in dicts[0].items():
#         if cnt >= 4:
#             imgidx0.append(idx)

#     for idx, cnt in dicts[1].items():
#         if cnt >= 4:
#             imgidx1.append(idx)
#     # 重建数据集
#     multi_train_set, _, criterion_dict = get_mix_balance_dataset(args, imgidx_PA100k=imgidx0, imgidx_DUKE=imgidx1)
#     train_loader = DataLoader(
#         dataset=multi_train_set,
#         batch_size=args.batchsize * 2,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=custom_collate
#     )
#     criterions = []
#     for dataset in args.dataset:
#         criterions.append(criterion_dict[dataset])

#     for i in range(46, epoch + 1):
#         scheduler.step(i)
#         train_loss, train_gt, train_probs = mix_batch_trainer(
#                 epoch=i,
#                 model=model,
#                 train_loader=train_loader,
#                 criterions=criterions,
#                 optimizer=optimizer,
#                 args=args,
#             )
        
#         for idx, dataset in enumerate(args.dataset):
#             valid_loader.dataset.init_set(dataset)
#             criterion = criterion_dict[dataset]
#             valid_loss, valid_gt, valid_probs = valid_trainer(
#                 model=model,
#                 valid_loader=valid_loader,
#                 criterion=criterion,
#             )
#             train_result = get_pedestrian_metrics(train_gt[idx], train_probs[idx])

#             print(f'Evaluation on train set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       train_result.ma, np.mean(train_result.label_pos_recall), np.mean(train_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
#                       train_result.instance_f1))
            
#             valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

#             print(f'Evaluation on test set:{dataset}, \n',
#                   'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#                       valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
#                   'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#                       valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
#                       valid_result.instance_f1))

#         print('-' * 60)
#         if i % args.epoch_save_ckpt == 0:
#             save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()