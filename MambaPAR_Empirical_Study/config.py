import argparse
import time

from tools.utils import str2bool


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP")
    
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=5)

    # 模型结构设置
    parser.add_argument('--only_img', action='store_true', default=False,help="Just use image brance")
    parser.add_argument('--no_VSF', action='store_true', default=False,help="don't use ASP mamba")
    parser.add_argument('--Text_not_train', action='store_true', default=False,help="don't use ASP mamba")
    parser.add_argument('--use_Trans', action='store_true', default=False,help="use Transformer block instead ASP mamba")
    parser.add_argument('--use_Vis_model', type=str, default="VMamba",help="use Vit as image encoder")
    parser.add_argument("--conv_dim", type=int, default=49,help="set last conv's in_channel when don't use ASP")
    parser.add_argument("--proj_text_dim", type=int, default=768,help="input text's dim")
    
    # Vim配置
    parser.add_argument('--Vim', default='vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='./checkpoints/Vim-small-midclstok/vim_s_midclstok_ft_81p6acc.pth',
                        type=str, metavar='CHECKPOINT',help='weight of model to load')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--nb_classes', default=26, type=int, help='dataset class num')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # 混合架构选择
    parser.add_argument('--hybrid', default=5, type=int, help='select hybrid')

    return parser
