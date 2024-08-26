
import os
import pprint
from collections import OrderedDict
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
from torch.cuda.amp import autocast
set_seed(605)
from tqdm import tqdm
from tools.utils import time_str
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
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=2,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )
    model = SeqPAR2(dataset=args.dataset, all_sentence=train_set.all_sentence, device=device,
                    attr_num=train_set.attr_num, attributes=train_set.attributes, lora_r=args.llama_lora_r,
                    max_txt_len=train_set.max_length, limit_words=train_set.limit_words) 
   
    if args.ckpt_path is not None:
        print(f'Loading The Backbone Model')
        checkpoint = torch.load(args.ckpt_path) 
        model.load_state_dict(checkpoint['updated_state_dict'],strict=False)
        
    model = model.to(device)
    model.eval()
    test_start=time.time()
    with torch.no_grad():
        gt_list = []
        mean_preds_probs = []
        for imgs, gt_label, imgname, img_simple_sentences,_ in tqdm(valid_loader):
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            with torch.cuda.amp.autocast(enabled=True):
                answer, logits_mean, logits_instance, llm_logits = model.infer(imgs, imgname, gt_label)
  
            m_logits = (logits_mean + logits_instance + llm_logits)/3.0    
            m_logits = torch.sigmoid(m_logits)
            mean_preds_probs.append(m_logits.detach().cpu().numpy())
        
        test_elapsed = time.time() - test_start
        print(f"Test in {test_elapsed // 60:.0f}m {test_elapsed % 60:.0f}s")
        gt_label = np.concatenate(gt_list, axis=0)
        mean_preds_probs = np.concatenate(mean_preds_probs, axis=0)
        mean_result = get_pedestrian_metrics(gt_label, mean_preds_probs)
        print(f'----------------------{time_str()} on Testing set----------------------\n',
            'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                mean_result.ma, np.mean(mean_result.label_pos_recall), np.mean(mean_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                mean_result.instance_acc, mean_result.instance_prec, mean_result.instance_recall,
                mean_result.instance_f1))
           
       

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()