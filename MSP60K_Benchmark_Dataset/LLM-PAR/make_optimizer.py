import torch


import torch
import torch.optim as optim

def make_optimizer(model, args, select_names=None, no_select_names=None):
    trainer_parameters = 0
    all_parameters = 0
    p_wd, p_non_wd = [], []
    
    for name, p in model.named_parameters():
        all_parameters += p.data.nelement()
        if not p.requires_grad:
            continue  # 跳过冻结的权重
        if select_names and not any(sel_name in name for sel_name in select_names):
            continue  # 当 select_name 存在且不在参数名中时跳过
        if no_select_names and any(no_sel_name in name for no_sel_name in no_select_names):
            continue  # 当 no_select_name 存在且在参数名中时跳过
        # print(name)
        if p.ndim < 2 or "bias" in name or "ln" in name or "bn" in name:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        trainer_parameters += p.data.nelement()
    
    print(f"trainable params: {trainer_parameters:d} || all params: {all_parameters:d}  || trainable%: {(trainer_parameters/all_parameters)*100:.6f}")
    print(f"Selected parameters with '{select_names}' and excluding '{no_select_names}'")
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(args.weight_decay),
        },
        {
            "params": p_non_wd,
            "weight_decay": 0,
        },
    ]
    
    optimizer = optim.AdamW(
        optim_params, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.999)
    )
    
    return optimizer
