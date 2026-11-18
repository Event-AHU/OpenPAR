import logging
import os
import pprint
import time
from pathlib import Path
from tools.checkpoint import adapt_lazy_visual_attention_state

def build_parameter_groups(model, args, stage, logger):
    keywords = args.update_parameters if stage == 'stage1' else args.update_parameters_stage2
    groups = []
    total_parameters = 0
    trainable_parameters = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        parameter.requires_grad = 'teacher' not in name and any((key in name for key in keywords))
        if parameter.requires_grad:
            logger.info('%s - %s', name, parameter.shape)
            groups.append({'params': [parameter], 'lr': args.lr, 'weight_decay': args.weight_decay})
            trainable_parameters += count
    logger.info('Total parameters: %.2fM', total_parameters / 1000000.0)
    logger.info('Trainable parameters: %.2fM (%.2f%%)', trainable_parameters / 1000000.0, 100 * trainable_parameters / total_parameters)
    return groups

def prepare_training_state(state, target_keys, args, stage):
    state = dict(state)
    if stage == 'stage2':
        state, converted, dropped = adapt_lazy_visual_attention_state(state, target_keys)
        print(f'Checkpoint attention keys: converted={converted}, dropped_aliases={dropped}')
        prefixes = ('group_sent_prompt_learner.',)
    else:
        prefixes = tuple(args.update_parameters)
    removed = [key for key in state if key.startswith(prefixes)]
    for key in removed:
        del state[key]
    print(f'Reinitialized parameters: {removed}')
    return state

def setup_logger(log_dir, timestamp):
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir / f'log_{timestamp}.txt')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def run_training(args, stage):
    import torch
    from torch.utils.data import DataLoader
    from batch_engine import batch_trainer, valid_trainer
    from dataset.OVAttrDataset import OpenVocabAttrDataset, get_transform
    from models.base_block_withSuperClass import GroupPAR
    from solver.scheduler_factory import make_scheduler
    from tools.utils import set_seed, time_str
    if stage == 'stage1':
        import cv2
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        cv2.setNumThreads(0)
    args.stage = stage
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus
    torch.cuda.set_device(int(args.vis_gpus))
    timestamp = time_str()
    log_dir = Path('logs') / args.dataset / args.exp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, timestamp)
    logger.info('Arguments:\n%s', pprint.pformat(vars(args)))
    train_transform, valid_transform = get_transform(args)
    train_set = OpenVocabAttrDataset(args=args, split=args.train_split, transform=train_transform)
    valid_set = OpenVocabAttrDataset(args=args, split=args.valid_split, transform=valid_transform)
    workers = 4 if stage == 'stage1' else 8
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batchsize * 4, shuffle=False, num_workers=workers, pin_memory=True)
    model = GroupPAR(train_set.base_attributes + train_set.novel_attributes, args, train_set, sample_weight=train_set.labels.mean(0)).cuda()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        state = prepare_training_state(state, model.state_dict().keys(), args, stage)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info('Missing keys: %s', missing)
        logger.info('Unexpected keys: %s', unexpected)
    groups = build_parameter_groups(model, args, stage, logger)
    optimizer = torch.optim.AdamW(groups, args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, num_epochs=args.epoch, lr=args.lr, warmup_t=10)
    start = time.time()
    for epoch in range(1, args.epoch + 1):
        scheduler.step(epoch)
        train_loss = batch_trainer(epoch=epoch, model=model, train_set=train_set, train_loader=train_loader, optimizer=optimizer, args=args, logger=logger)
        logger.info('Evaluation on training set, train_loss: %.4f', train_loss)
        predictions, novel_predictions, labels, novel_labels = valid_trainer(epoch=epoch, model=model, val_set=valid_set, valid_loader=valid_loader, args=args)
        print_metric(predictions, labels, novel_predictions, novel_labels, logger)
        if epoch % args.save_freq == 0:
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict(), 'model_state_dict': model.state_dict()}, log_dir / f'epoch{epoch}.pth')
        if epoch % 5 == 0:
            logger.info('Time for %d epochs: %.2f seconds', epoch, time.time() - start)

def print_metric(preds_probs, gt_label, novel_preds_probs, novel_gt_label, logger):
    import torch
    from misc import compute_AP, compute_F1
    preds_probs = torch.tensor(preds_probs).cuda()
    gt_label = torch.tensor(gt_label).cuda()
    novel_preds_probs = torch.tensor(novel_preds_probs).cuda()
    novel_gt_label = torch.tensor(novel_gt_label).cuda()
    all_preds_probs = preds_probs
    all_gt_label = gt_label
    logger.info('completed calculating predictions over all images')
    logits_novel_5 = novel_preds_probs.clone()
    ap_Novel = compute_AP(novel_preds_probs, novel_gt_label)
    F1_3_Novel, P_3_Novel, R_3_Novel = compute_F1(novel_preds_probs, novel_gt_label, 'overall', k_val=3)
    F1_5_Novel, P_5_Novel, R_5_Novel = compute_F1(logits_novel_5, novel_gt_label, 'overall', k_val=5)
    logger.info('ZSL AP: {:.4f}'.format(torch.mean(ap_Novel)))
    logger.info('k=3: F1: {:.4f}, P: {:.4f}, R: {:.4f}'.format(torch.mean(F1_3_Novel), torch.mean(P_3_Novel), torch.mean(R_3_Novel)))
    logger.info('k=5: F1: {:.4f}, P: {:.4f}, R: {:.4f}'.format(torch.mean(F1_5_Novel), torch.mean(P_5_Novel), torch.mean(R_5_Novel)))
    logits_all_5 = all_preds_probs.clone()
    ap_Total = compute_AP(all_preds_probs, all_gt_label)
    F1_3_Total, P_3_Total, R_3_Total = compute_F1(all_preds_probs, all_gt_label, 'overall', k_val=3)
    F1_5_Total, P_5_Total, R_5_Total = compute_F1(logits_all_5, all_gt_label, 'overall', k_val=5)
    logger.info('GZSL AP: {:.4f}'.format(torch.mean(ap_Total)))
    logger.info('g_k=3: F1: {:.4f}, P: {:.4f}, R: {:.4f}'.format(torch.mean(F1_3_Total), torch.mean(P_3_Total), torch.mean(R_3_Total)))
    logger.info('g_k=5: F1: {:.4f}, P: {:.4f}, R: {:.4f}'.format(torch.mean(F1_5_Total), torch.mean(P_5_Total), torch.mean(R_5_Total)))
    logger.info('-' * 60)
