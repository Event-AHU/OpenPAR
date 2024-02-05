import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP1")
    
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="AdamW", choices=['SGD', 'Adam', 'AdamW'])

    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='6', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=1)

    parser.add_argument("--check_point", action='store_true')
    parser.add_argument("--dir", type=str, default=None)
    #beam_search 参数
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--beam_size", type=int, default=1)
    
    parser.add_argument("--pkl_path", type=str, default='/wangxiao/jjd/model/peta.pkl')
    parser.add_argument("--datasets_path", type=str, default='/wangxiao/jjd/PETAPad/Pad_datasets')
    parser.add_argument("--model_path", type=str, default='/wangxiao/jjd/model')
    parser.add_argument("--use_class_weight", action='store_true', help='use dual-coop instead of prompt')
    return parser
