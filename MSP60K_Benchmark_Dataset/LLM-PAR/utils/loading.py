import torch
import os 

def loading_only_update(model, ckpt_path, optimizer, optimizer_llm) :
    checkpoint = torch.load(ckpt_path) 
    epoch = checkpoint['epoch']
    print(f'Loading {epoch}-st Epoch Pretrain Model from {ckpt_path}')
    model_state_dict = model.load_state_dict(checkpoint['updated_state_dict'],strict=False)
    optimizer.load_state_dict(checkpoint['base_state_dict'])
    optimizer_llm.load_state_dict(checkpoint['llm_state_dict'])
    
    # for name, param in checkpoint['updated_state_dict'].items():
    #     if name in model_state_dict :
    #         # print(f'Loading Param {name} Over !!')
    #         model_state_dict[name].copy_(param)
    #     else:
    #         print("Warning: Ignoring missing key:", name)   
        
    return epoch
            
def save_model_only_update(model_state_dict, loss, epoch, file_name):
    torch.save({
        'updated_state_dict': {name: param.detach().cpu() for name, param in model_state_dict.items() if param.requires_grad},
        'loss': loss,
        'epoch': epoch,
    }, file_name)
    
import sys
import time

class ProgressBar:
    def __init__(self, total_steps, total_prints=10, bar_length=40, head="Progress"):
        self.total_steps = total_steps
        self.total_prints = total_prints
        self.bar_length = bar_length
        self.start_time = time.time()
        self.head = head
    def print_progress_bar(self, step):
        interval = max(1, self.total_steps // self.total_prints)
        if step % interval == 0 or step == self.total_steps:
            progress = step / self.total_steps
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time

            arrow = '-' * int(progress * self.bar_length - 1) + '>'
            spaces = ' ' * (self.bar_length - len(arrow))

            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'

            sys.stdout.write(f'\r{self.head}: [{arrow}{spaces}] {int(progress * 100)}% | Remaining time: {time_str}')
            sys.stdout.flush()

    def reset(self):
        self.start_time = time.time()