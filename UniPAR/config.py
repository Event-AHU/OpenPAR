import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument("--dataset", type=str, default=['DUKE','EventPAR'])
    parser.add_argument("--lossrate", type=str, default=[1,1,1]) #0.8 ,1,0.5
    # parser.add_argument("--dataset", type=str, default=['PA100k', 'DUKE'])
    # parser.add_argument("--dataset", type=str, default=['MSP60k'])
    parser.add_argument("--dataset", type=str, default=['MSP60k','DUKE','EventPAR'])
    # parser.add_argument("--lossrate", type=str, default=[0.8,0.5]) #0.8 ,1,0.5
    parser.add_argument("--save_place", type=str, default="multiDataset")
    
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='5', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=20)

    parser.add_argument("--PE_load", action='store_true')
    parser.add_argument("--accumulate", action='store_true')
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--union_token", action='store_true')
    parser.add_argument("--islossrate", action='store_true')
    parser.add_argument("--repet_DUKE", type=int, default=1)
    parser.add_argument('--visual_only', action='store_true')
    return parser
