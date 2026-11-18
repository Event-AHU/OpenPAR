import csv
import json
import os
from pathlib import Path
from config import argument_parser

def align_sentence_prompt_mode(args, state):
    context = state.get('group_sent_prompt_learner.ctx')
    if context is not None:
        if context.ndim != 2:
            raise ValueError('Sentence-prompt context must be a two-dimensional tensor')
        args.no_sen_prompt = context.shape[0] == 0
        if context.shape[0]:
            args.context_len = int(context.shape[0])

def run_evaluation(args):
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise FileNotFoundError('Set --checkpoint to a trained Stage-2 checkpoint')
    import torch
    from torch.utils.data import DataLoader
    from batch_engine import valid_trainer
    from dataset.OVAttrDataset import OpenVocabAttrDataset, get_transform
    from misc import compute_metrics
    from models.base_block_withSuperClass import GroupPAR
    from tools.checkpoint import adapt_lazy_visual_attention_state
    from tools.utils import set_seed
    args.stage = 'stage2'
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus
    torch.cuda.set_device(int(args.vis_gpus))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    align_sentence_prompt_mode(args, state)
    train_transform, valid_transform = get_transform(args)
    train_set = OpenVocabAttrDataset(args=args, split=args.train_split, transform=train_transform)
    valid_set = OpenVocabAttrDataset(args=args, split=args.valid_split, transform=valid_transform)
    loader = DataLoader(valid_set, batch_size=args.batchsize * 4, shuffle=False, num_workers=4, pin_memory=True)
    model = GroupPAR(train_set.base_attributes + train_set.novel_attributes, args, train_set, sample_weight=train_set.labels.mean(0)).cuda()
    state, converted, dropped = adapt_lazy_visual_attention_state(state, model.state_dict().keys())
    print(f'Checkpoint attention keys: converted={converted}, dropped_aliases={dropped}')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Missing keys: {missing}')
    print(f'Unexpected keys: {unexpected}')
    predictions, novel_predictions, labels, novel_labels = valid_trainer(epoch=0, model=model, val_set=valid_set, valid_loader=loader, args=args)
    metrics = compute_metrics(torch.from_numpy(predictions), torch.from_numpy(labels), torch.from_numpy(novel_predictions), torch.from_numpy(novel_labels), base_count=len(valid_set.base_indices))
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {'dataset': args.dataset, 'checkpoint': str(args.checkpoint), 'metrics': metrics, 'config': vars(args), 'missing_keys': list(missing), 'unexpected_keys': list(unexpected)}
    result_path = output / f'{args.exp}.json'
    with result_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    csv_path = output / f'{args.exp}.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'checkpoint', *metrics])
        writer.writeheader()
        writer.writerow({'dataset': args.dataset, 'checkpoint': str(args.checkpoint), **metrics})
    print('Metrics (%)')
    for name, value in metrics.items():
        print(f'  {name}: {100 * value:.2f}')
    print(f'Results: {result_path} and {csv_path}')
    return metrics
if __name__ == '__main__':
    run_evaluation(argument_parser())
