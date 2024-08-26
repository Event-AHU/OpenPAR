
import os
import pprint
from collections import OrderedDict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.attr2vec_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str,set_seed
from make_optimizer import make_optimizer
from loss.CE_loss import CEL_Sigmoid
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast
from utils.train_utils import *
from utils.loading import loading_only_update

set_seed(605)
def main(args):
    log_dir = os.path.join('logs', args.dataset, args.exp)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        
    train_tsfm, valid_tsfm = get_transform(args) #数据增强
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) #得到训练数据集
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)  # 验证集 validation
     
    start_time=time_str()
    print(f'start_time is {start_time}')
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')
        
        
    device = torch.device("cuda") 
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        num_workers=8,
        pin_memory=True,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize*6,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )
    labels = train_set.label
    sample_weight = labels.mean(0)   
    
    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    criterion_llm = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    model = SeqPAR2(dataset=args.dataset, all_sentence=train_set.all_sentence, device=device,attr_num=train_set.attr_num, 
                    cross_layer_num = args.cross_layers, num_query = args.num_query,
                    attributes=train_set.attributes, lora_r=args.llama_lora_r,
                    max_txt_len=train_set.max_length, limit_words=train_set.limit_words) 
    model = model.to(device)
    start_epoch = 0


    optimizer = make_optimizer(model, args, no_select_names=['llm','llama_proj','llama_model','query_tokens','Qformer'])
    optimizer_llm = make_optimizer(model, args, select_names=['llm','llama_proj','llama_model','query_tokens','Qformer'])
    get_parameters(model)
    if args.stage1_ckpt_path is not None:
        print(f'Loading The Backbone Model')
        start_epoch = loading_only_update(model, args.stage1_ckpt_path, 
                                          optimizer, optimizer_llm)
      
        
    lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=(len(train_loader) * args.epoch)
        )
    lr_scheduler_llm = get_linear_schedule_with_warmup(
            optimizer=optimizer_llm,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=(len(train_loader) * args.epoch)
        )
    trainer(args=args,
            epoch=args.epoch,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer={'base':optimizer, 'llm':optimizer_llm},
            lr_scheduler={'base':lr_scheduler, 'llm':lr_scheduler_llm},
            criterion=criterion,
            criterion_llm = criterion_llm,
            path=log_dir,
            min_lr=args.min_lr,
            start_epoch=start_epoch
            )
def trainer(args, epoch, model, train_loader, valid_loader, optimizer, lr_scheduler, criterion, criterion_llm, path, min_lr,start_epoch):
    start=time.time()
    metric_logger = MetricLogger(delimiter="  ")
    # 新建log.txt文件
    log_file_path = os.path.join(path, "log.txt")
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write("Epoch\tLoss\t\tmA\t\tF1\n")  # 写入表头
    
    for i in range(1, start_epoch+1):
        print(f"{i}-th Epoch LR Advise")
        for step, (imgs, gt_label, imgname, img_simple_sentences, random_sentences) in enumerate(train_loader):
            for lrs_name, lrs in lr_scheduler.items():
                lrs.step()
                
    for i in range(start_epoch+1, epoch + 1):
        batch_num = len(train_loader)
        header = 'Epoch: [{}]'.format(i)
        gt_list = []
        mean_preds_probs = []
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        for step, (imgs, gt_label, imgname, img_simple_sentences, random_sentences) in enumerate(metric_logger.log_every(train_loader, int(batch_num / 10), header)):
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            for lrs_name, lrs in lr_scheduler.items():
                lrs.step()
            for opt_name, opt in optimizer.items():
                for param_group in opt.param_groups:
                    # 检查当前学习率是否低于最小学习率，如果是则设置为最小学习率
                    if param_group["lr"] < min_lr:
                        param_group["lr"] = min_lr

            with torch.cuda.amp.autocast(enabled=True):
                llm_loss, logits_mean, logits_instance, llm_logits = model(imgs, img_simple_sentences, random_sentences, gt_label)
                
                cls_mean_loss = criterion(logits_mean, gt_label.clone())
                cls_in_loss = criterion(logits_instance, gt_label.clone())
                cls_llm_loss = criterion(llm_logits, gt_label.clone())
                cls_loss = (cls_mean_loss + cls_in_loss)/2.0
                cls_llm_loss = cls_llm_loss + llm_loss
                
            scaler.scale(cls_loss).backward()
            scaler.step(optimizer['base'])
            scaler.update()
            optimizer['base'].zero_grad()
            optimizer['llm'].zero_grad()
            
            scaler.scale(cls_llm_loss).backward()
            scaler.step(optimizer['llm'])
            scaler.update()
            optimizer['base'].zero_grad()
            optimizer['llm'].zero_grad()
            
            m_logits = (logits_mean + logits_instance + llm_logits)/3.0
            m_logits = torch.sigmoid(m_logits)
            
            mean_preds_probs.append(m_logits.detach().cpu().numpy())           
            gt_list.append(gt_label.cpu().detach().numpy())

            metric_logger.update(loss=cls_llm_loss.item())
            metric_logger.update(Ll=llm_loss.item())
            metric_logger.update(lr=optimizer['base'].param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        gt_label = np.concatenate(gt_list, axis=0)
        mean_preds_probs = np.concatenate(mean_preds_probs, axis=0)
        mean_result = get_pedestrian_metrics(gt_label, mean_preds_probs)
        print('Mean Result: ma: {:.4f},  F1: {:.4f}'.format(mean_result.ma, mean_result.instance_f1))

        if i % args.epoch_save_ckpt==0 and i>=5:
            updated_state_dict={}
            # Save selected parameters
            for name, param in model.named_parameters():
                if param.requires_grad == True or 'bn' in name:
                    updated_state_dict[name] = param.detach().cpu()
            updated_state_dict['bn_class.running_mean'] = model.bn_class.running_mean        
            updated_state_dict['bn_class.running_var'] = model.bn_class.running_var
            updated_state_dict['bn_llm.running_mean'] = model.bn_llm.running_mean        
            updated_state_dict['bn_llm.running_var'] = model.bn_llm.running_var
            updated_state_dict['bn_instance.running_mean'] = model.bn_instance.running_mean        
            updated_state_dict['bn_instance.running_var'] = model.bn_instance.running_var    
            torch.save({
                'updated_state_dict': updated_state_dict,
                'base_state_dict':optimizer['base'].state_dict(),
                'llm_state_dict':optimizer['llm'].state_dict(),
                'epoch': i,
                    # 如果有其他需要保存的内容，也可以添加到这里
            }, os.path.join(path, f"Epoch_{str(i)}.pth")) 
            
            
        if i % args.eval_freq==0 and i>5:
            answer_file_path = os.path.join(path, f"infer_sentences_epoch_{i}.txt")
            test_start=time.time()
            with torch.no_grad():
                model.eval()
                batch_num = len(valid_loader)
                gt_list = []
                mean_preds_probs = []
                img_names=[]
                infer_sentences=[]
                gt_sentences=[]
                for step, (imgs, gt_label, imgname, img_simple_sentences, random_sentences) in enumerate(valid_loader):
                    imgs, gt_label = imgs.cuda(), gt_label.cuda()
                    with torch.cuda.amp.autocast(enabled=True):
                        answers, logits_mean, logits_instance, llm_logits = model.infer(imgs, gt_label)
                        cls_mean_loss = criterion(logits_mean, gt_label)
                        cls_in_loss = criterion(logits_instance, gt_label)
                        cls_llm_loss = criterion(llm_logits, gt_label)
                        loss = (cls_mean_loss + cls_in_loss + cls_llm_loss)/3.0
                        
                    infer_sentences=infer_sentences+answers
                    gt_sentences=gt_sentences+img_simple_sentences
                    img_names = img_names+imgname
                    
                    m_logits = (logits_mean + logits_instance + llm_logits)/3.0
                    m_logits = torch.sigmoid(m_logits)
                    
                    mean_preds_probs.append(m_logits.detach().cpu().numpy())
                    gt_list.append(gt_label.cpu().detach().numpy())

                    
            with open(answer_file_path, 'w') as f:
                for idx in range(len(infer_sentences)):
                    f.write(f'图片：{img_names[idx]}\n') 
                    f.write(f'预测：\n{infer_sentences[idx]}\n') 
                    f.write(f'真值：\n{gt_sentences[idx]}\n\n')    
                      
            test_elapsed = time.time() - test_start
            print(f"Test in {test_elapsed // 60:.0f}m {test_elapsed % 60:.0f}s")
            gt_label = np.concatenate(gt_list, axis=0)
            
            mean_preds_probs = np.concatenate(mean_preds_probs, axis=0)
            mean_result = get_pedestrian_metrics(gt_label, mean_preds_probs)
            cur_epoch_loss = loss.item()
            print(f'----------------------{time_str()} on Testing set Mean Result, Loss:{cur_epoch_loss:.4f}----------------------\n',
                'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                    mean_result.ma, np.mean(mean_result.label_pos_recall), np.mean(mean_result.label_neg_recall)),
                'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    mean_result.instance_acc, mean_result.instance_prec, mean_result.instance_recall,
                    mean_result.instance_f1))
           
            print('-' * 60)     
    elapsed = time.time() - start
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    
def get_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = []
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            trained_params.append(param)
    trained_params_count = sum(p.numel() for p in trained_params)
    trainable_percentage = ((trained_params_count) / total_params) * 100 if total_params > 0 else 0
    print(f"trainable params: {(trained_params_count)} || all params: {total_params} || trainable%: {trainable_percentage:.6f}")
        
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()