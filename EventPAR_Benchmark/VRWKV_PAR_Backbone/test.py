# eval.py
import os
import pprint
from collections import OrderedDict
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import TransformerClassifier
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, set_seed

from batch_engine import valid_trainer

set_seed(605)





def main(args):
    print("=" * 60)
    print("               EVALUATION SCRIPT START")
    print("=" * 60)
    print(time_str())
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60)

    # --------------------------------------------------
    #  Dataset / Dataloader
    # --------------------------------------------------
    
    train_tsfm, valid_tsfm = get_transform(args)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    sample_weight = train_set.label.mean(0)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize if args.batchsize > 0 else 16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Valid dataset size: {len(valid_loader.dataset)}  |  Attr num: {train_set.attr_num}")
    print("Attributes:")
    print(",".join([str(a) for a in train_set.attributes]))

    # --------------------------------------------------
    #  Build Model
    # --------------------------------------------------
    print("\nBuilding model...")
    model = TransformerClassifier(train_set.attr_num)

    # --------------------------------------------------
    #  Load Checkpoint
    # --------------------------------------------------
    
    checkpoint= torch.load(args.dir)
    model.load_state_dict(checkpoint['state_dicts'],strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)

    # --------------------------------------------------
    #  Evaluation
    # --------------------------------------------------
    print("\nStart Evaluation ...")
    start = time.time()

    valid_loss, valid_gt, valid_probs = valid_trainer(
        model=model,
        valid_loader=valid_loader,
        criterion=criterion,
    )

    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

    # --------------------------------------------------
    #  Print Results
    # --------------------------------------------------
    print("\n======= Evaluation Result =======")
    print(f'valid_loss: {valid_loss:.4f}')
    print('ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
        valid_result.ma,
        np.mean(valid_result.label_pos_recall),
        np.mean(valid_result.label_neg_recall)
    ))
    print('Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
        valid_result.instance_acc,
        valid_result.instance_prec,
        valid_result.instance_recall,
        valid_result.instance_f1,
    ))

    total = time.time() - start
    print(f"\nTotal evaluation time: {total:.2f} sec")
    print("=" * 60)


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
