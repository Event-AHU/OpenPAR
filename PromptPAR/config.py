import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP1")
    
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--use_div", action='store_true')
    parser.add_argument("--use_vismask", action='store_true')
    parser.add_argument("--use_GL", action='store_true')
    parser.add_argument("--ag_threshold", type=float, default=0.5)
    parser.add_argument("--smooth_param", type=float, default=0.1)
    parser.add_argument("--use_textprompt", action='store_true')
    parser.add_argument("--use_mm_former", action='store_true')
    parser.add_argument("--mm_layers", type=int, default=1)
    
    parser.add_argument("--div_num", type=int, default=4)
    parser.add_argument("--overlap_row", type=int, default=2)
    parser.add_argument("--text_prompt", type=int, default=3)
    parser.add_argument("--vis_prompt", type=int, default=25)
    parser.add_argument("--vis_depth", type=int, default=24)
    parser.add_argument("--clip_lr", type=float, default=4e-3)
    parser.add_argument("--clip_weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--mmformer_update_parameters", type=list, default=["word_embed", "weight_layer", "bn", "norm"])
    parser.add_argument("--clip_update_parameters", type=list, default=["prompt_deep", "prompt_text_deep", "part_class_embedding", "agg_bn", "softmax_model"])
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--save_freq", type=int, default=1)

    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--dir", type=str, default=None)
    return parser