import pprint
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from eval_batch import valid_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,get_signle_metrics
from tools.utils import time_str, set_seed
from collections import Counter
from torchtext.vocab import vocab
from CLIP.model import *
set_seed(605)
device = "cuda"
def main(args):

    if args.check_point==False :
        print(time_str())
        pprint.pprint(OrderedDict(args.__dict__))
        print('-' * 60)
        print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args) 
    train_tsfm, valid_tsfm = get_transform(args) 
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) 
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 
    args.batchsize=1
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    if args.use_TextPrompt :
        print('Use Continuous Prompt!')
    else :
        print('Use Handcrafted Prompt!')
    counter_attr = Counter()
    counter_attr.update(train_set.attributes)
    vocab_attr = vocab(counter_attr, min_freq=1, specials=( '<bos>', '<eos>', '<pad>'))
    base_index2attr=dict(zip([i for i in range(len(train_set.attributes))],train_set.attributes))
    labels = train_set.label
    sample_weight = labels.mean(0)
    model = TransformerClassifier(train_set.attr_num,train_set.attributes,vocab_attr,args)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.check_point :
        print("start loading decoder model")
        checkpoint = torch.load(args.dir)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        epoch = checkpoint['epoch']
        valid_ma = checkpoint['valid_ma']
        valid_f1 = checkpoint['valid_f1']
        print(f"loading decoder model over, model is epoch{epoch},valid_ma is {valid_ma:.4f},valid_f1 is {valid_f1:.4f}")   
    if torch.cuda.is_available():
        model = model.cuda()
    trainer(model=model,
            valid_loader=valid_loader,
            vocab_attr=vocab_attr,
            base_index2attr=base_index2attr,
            attributes=valid_set.attributes,
            args=args)
    
def trainer(model,valid_loader,vocab_attr,base_index2attr,attributes,args):
    start=time.time()
    valid_gt, valid_probs,valid_loss,imgnames,pred_attrs,gt_attrs = valid_trainer(
        model=model,
        valid_loader=valid_loader,
        vocab_attr=vocab_attr,
        base_index2attr=base_index2attr,
        attributes=attributes,
        args=args
        )
    ma = []
    acc = []
    f1 = []

    img_attrs=dict(zip(imgnames,pred_attrs))
    gt_img_attrs=dict(zip(imgnames,gt_attrs))
    # 打开一个文本文件以写入模式
    with open('img_attrs.txt', 'w') as file:
        # 遍历字典的键值对，按行写入文本文件
        for key, value in img_attrs.items():
            file.write(f'{key}: {value}\n')   
    # 打开一个文本文件以写入模式
    with open('gt_img_attrs.txt', 'w') as file:
        # 遍历字典的键值对，按行写入文本文件
        for key, value in gt_img_attrs.items():
            file.write(f'{key}: {value}\n')              
    # 计算每行的差异
    row_differences = np.abs(valid_gt - valid_probs).sum(axis=1)

    # 找出行相同的索引
    same_rows_indices = np.where(row_differences == 0)[0]
    pred_right_name=[imgnames[int(elem)] for elem in  same_rows_indices]
    # 找出行差距最大的前10个索引
    max_diff_indices = np.argsort(row_differences)[-10:][::-1]
    pred_bad_name=[imgnames[int(elem)] for elem in  max_diff_indices]
    print("完全预测正确的：",pred_right_name)

    print("行差距最大的前10个索引：", pred_bad_name)

    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
    for i in range(35):
        result = get_signle_metrics(valid_gt[:,i], valid_probs[:,i])
        ma.append(result.ma)
        acc.append(result.instance_acc)
        f1.append(result.instance_f1)
    print(ma)
    print(acc)
    print(f1)
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