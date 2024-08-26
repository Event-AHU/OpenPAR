import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="PETA")
    
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--eval_start", type=int, default=5)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cross_layers", type=int, default=3)
    parser.add_argument("--num_query", type=int, default=128)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    
    parser.add_argument('--gpus', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=1)
    parser.add_argument("--v_lora_r", type=int, default=32)
    parser.add_argument("--v_lora_alpha", type=int, default=16)
    parser.add_argument("--v_lora_dropout", type=float, default=0.05)
    parser.add_argument("--v_lora_target_modules", type=list, default=["q","v"])
    parser.add_argument("--llama_lora_r", type=int, default=32)
    parser.add_argument("--llama_lora_alpha", type=int, default=16)
    parser.add_argument("--llama_lora_dropout", type=float, default=0.05)
    parser.add_argument("--llama_lora_target_modules", type=list, default=["q_proj", "v_proj"])

    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument("--only_testing", action='store_true')
    return parser