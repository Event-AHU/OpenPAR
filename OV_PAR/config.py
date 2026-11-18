import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def argument_parser(argv=None):
    parser = argparse.ArgumentParser(description='SRENet: complete model, standard MSPOV/RAPOV2 protocols', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=['MSPOV', 'RAPOV2'])
    parser.add_argument('--dataset_pkl', default=None)
    parser.add_argument('--image_root', default=None)
    parser.add_argument('--semantic_bank_path', default=None)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--vis_gpus', default='0', help='GPU index used by the original training entry points')
    parser.add_argument('--exp', default='srenet')
    parser.add_argument('--train_split', default='train', choices=['train', 'trainval'])
    parser.add_argument('--valid_split', default='test', choices=['test', 'valid'])
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--stage', choices=['stage1', 'stage2'], default='stage1')
    parser.add_argument('--context_len', type=int, default=32)
    parser.add_argument('--vp_len', type=int, default=None)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--adapter_h', type=int, default=8)
    parser.add_argument('--results_dir', default=str(ROOT / 'results'))
    args = parser.parse_args(argv)
    is_mspov = args.dataset == 'MSPOV'
    if args.dataset_pkl is None:
        args.dataset_pkl = str(ROOT / 'dataset' / ('msp_ov.pkl' if is_mspov else 'rap_ov2.pkl'))
    if args.image_root is None:
        args.image_root = str(ROOT / 'data' / ('MSP60K/images' if is_mspov else 'RAPv2/RAP_dataset'))
    if args.semantic_bank_path is None:
        bank = 'instruction/mspov_basetrain_novelall_group_combos.json' if is_mspov else 'instruction/2/rapov2_basetrain_novelall_group_combos.json'
        args.semantic_bank_path = str(ROOT / bank)
    if args.vp_len is None:
        args.vp_len = 10 if is_mspov else 15
    args.no_sen_prompt = is_mspov
    args.update_parameters = ['crossattn', 'visual_adapters', 'text_guide_gate', 'prompt', 'global_head', 'temperature']
    args.update_parameters_stage2 = ['prompt_learner']
    return args
