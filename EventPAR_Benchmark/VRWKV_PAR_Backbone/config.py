import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="EventPAR")
    
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_t", type=int, default=5) 
    parser.add_argument("--optim", type=str, default="SGD", choices=['SGD', 'Adam', 'AdamW'])

    parser.add_argument("--train_split", type=str, default="train", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='6', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=1)
    parser.add_argument('--rwkv_config', default='models/vrwkv_configs/vrwkv6/vrwkv6_base_16xb64_in1k.py', help='config file path')
    parser.add_argument('--rwkv_checkpoint', default='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/wanghaiyang/baseline/VRWKV_PAR_Backbone/vrwkv6_b_in1k_224.pth', help='checkpoint file')
    
    parser.add_argument("--multi", action='store_true')

    parser.add_argument("--fusion_mode", type=str, default="concat", choices=['concat', 'add', 'conv'])
    parser.add_argument("--backbones", type=str, default="rwkv", choices=['rwkv', 'vit', 'resnet50'])
    return parser