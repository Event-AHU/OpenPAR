import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP")
    
    parser.add_argument("--batchsize", type=int, default=12)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=5)
    parser.add_argument("--dir", type=str, default="your checkpoint path")
    parser.add_argument("--temp", type=int, default=2)


    parser.add_argument("--only_feats_kl", action='store_true')
    parser.add_argument("--only_logits_kl", action='store_true')
    
    return parser
