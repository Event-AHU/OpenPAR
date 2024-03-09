import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn,optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
from collections import Counter
from torchtext.vocab import vocab
from decoder.transformer import CaptionTransformer
from CLIP.clip import clip
from CLIP.clip.model import *
from decoder.decoders import DecoderLayer,Decoder
from tensorboardX import SummaryWriter
from loss.NLL_loss import NLL_class_weight_loss
set_seed(605)
device = "cuda"
#选择image特征提取器
def main(args):
    start_time=time_str()
    print(f'start_time is {start_time}')
    tb_writer = SummaryWriter('tensorboardX/exp')
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

    if args.check_point==False :
        print(time_str())
        pprint.pprint(OrderedDict(args.__dict__))
        print('-' * 60)
        #select_gpus(args.gpus)
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
    counter_attr = Counter()
    counter_attr.update(train_set.attributes)
    vocab_attr = vocab(counter_attr, min_freq=1, specials=( '<bos>', '<eos>', '<pad>'))
    base_index2attr=dict(zip([i for i in range(len(train_set.attributes))],train_set.attributes))
    labels = train_set.label
    sample_weight = labels.mean(0)
    model = TransformerClassifier(train_set.attr_num,train_set.attributes,vocab_attr,args)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = NLL_class_weight_loss(sample_weight, attr_idx=train_set.attr_num)
    lr = args.lr
    epoch_num = args.epoch
    start_epoch=1
    if args.optim == 'SGD' :
        optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
        print('The optimizer used SGD')
    elif args.optim == 'AdamW' :
        optimizer = optim.AdamW(model.parameters(),lr=lr, weight_decay=args.weight_decay)
        print('The optimizer used AdamW')
    else :
        optimizer = optim.Adam([{'params': model.decoder.parameters(), 'lr': args.lr},{'params': model.ViT_model.parameters(), 'lr': args.lr}])
        print('The optimizer used Adam')
    if args.use_class_weight :
        print('The loss function used NLL_loss with class weight')
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)
    if args.check_point :
        print("start loading decoder model")
        checkpoint = torch.load(args.dir)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        start_epoch = checkpoint['epoch']+1
        #valid_loss = checkpoint['valid_loss']
        valid_ma = checkpoint['valid_ma']
        valid_f1 = checkpoint['valid_f1']
        print(f"loading decoder model over, from epoch{start_epoch},valid_ma:{valid_ma},valid_f1:{valid_f1},")    
    trainer(epoch=epoch_num,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            path=log_dir,
            tb_writer=tb_writer,
            start_epoch=start_epoch,
            vocab_attr=vocab_attr,
            base_index2attr=base_index2attr,
            attributes=train_set.attributes,
            args=args)
    
def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, scheduler,path,tb_writer,start_epoch,vocab_attr,base_index2attr,attributes,args):
    max_ma,min_loss,max_f1=0,0,0
    max_ma_epoch,max_f1_epoch = 0,0
    start=time.time()
    for i in range(start_epoch, epoch+1):
        scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocab_attr=vocab_attr,
            base_index2attr=base_index2attr,
            attributes=attributes,
            args=args
        )
        #print train
        train_result = get_pedestrian_metrics(train_gt, train_probs)
        print(f'----------------------{time_str()} on train set----------------------\n',
              'ma: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(
                  train_result.ma, train_result.instance_acc, train_result.instance_f1))  
        if train_result.ma > 0.5:
            valid_gt, valid_probs ,valid_loss = valid_trainer(
                model=model,
                valid_loader=valid_loader,
                vocab_attr=vocab_attr,
                base_index2attr=base_index2attr,
                attributes=attributes,
                args=args
            )
            valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
            print(f'----------------------{time_str()} on Evalution set,valid_loss:----------------------\n',
                'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                    valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
                'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                    valid_result.instance_f1))                 
            print('-' * 60)   

            tb_writer.add_scalar("valid_mA", valid_result.ma, i )
            tb_writer.add_scalar("valid_Acc", valid_result.instance_acc, i )
            tb_writer.add_scalar("valid_F1", valid_result.instance_f1, i )
            if max_ma < valid_result.ma :
                max_ma = valid_result.ma
                max_ma_epoch = i
                torch.save({
                            'epoch': i,
                            'model_state_dict': model.state_dict(),
                            'valid_ma':valid_result.ma,
                            'valid_f1':valid_result.instance_f1
                            }, os.path.join(path, f"ma_best.pth")) 
            if max_f1 < valid_result.instance_f1 :
                max_f1 = valid_result.instance_f1
                max_f1_epoch = i
                torch.save({
                            'epoch': i,
                            'model_state_dict': model.state_dict(),
                            'valid_ma':valid_result.ma,
                            'valid_f1':valid_result.instance_f1
                            }, os.path.join(path, f"f1_best.pth"))
            print('best ma is {:.4f}, in epoch {} best f1 is {:.4f}, in epoch {}'.format(max_ma,max_ma_epoch,max_f1,max_f1_epoch))    

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
