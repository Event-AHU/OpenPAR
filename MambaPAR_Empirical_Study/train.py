import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


def main(args):
    if args.use_Vis_model != "Vmamba":
        args.conv_dim = 197

    log_dir = os.path.join('logs', args.dataset+"_"+args.use_Vis_model)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    visenv_name = args.dataset
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

    model = MambaClassifier(args,train_set.attr_num,train_set.words)
    print(model)
    
    if torch.cuda.is_available():
        model = model.cuda()

    for name, param in model.named_parameters():
        print(name,param.requires_grad)

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    lr = args.lr
    epoch_num = args.epoch

    optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)

    best_metric, epoch = trainer(epoch=epoch_num,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir)
    print(f'{visenv_name},  best_ma : {best_metric[0]} in epoch{epoch[0]}')
    print(f'{visenv_name},  best_acc: {best_metric[1]} in epoch{epoch[1]}')
    print(f'{visenv_name},  best_f1 : {best_metric[2]} in epoch{epoch[2]}')
    
def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, scheduler, path):
    maximum1, maximum2, maximum3 = float(-np.inf), float(-np.inf),float(-np.inf)
    best_epoch1, best_epoch2, best_epoch3 = 0,0,0

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
        
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print('-' * 60)
        # if i % args.epoch_save_ckpt:
        #     save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)
        cur_metric1 = valid_result.ma
        cur_metric2 = valid_result.instance_acc
        cur_metric3 = valid_result.instance_f1
        if cur_metric1 > maximum1:
            maximum1 = cur_metric1
            best_epoch1 = i
            save_ckpt(model, os.path.join(path, 'ckpt_best_Ma.pth'), i, maximum1)
            # save_ckpt(model, '/home/zhexuan_wh/model/m.pth', i, maximum1)
            # torch.save({'pt': valid_probs, 'gt':valid_gt}, '/home/zhexuan_wh/model/result.pkl')
        
        if cur_metric2 > maximum2:
            maximum2 = cur_metric2
            best_epoch2 = i
        if cur_metric3 > maximum3:
            maximum3 = cur_metric3
            best_epoch3 = i
            save_ckpt(model, os.path.join(path, 'ckpt_best_F1.pth'), i, maximum1)
    
    return (maximum1,maximum2,maximum3), (best_epoch1,best_epoch2,best_epoch3)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
